import json
import os
import sys
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, ".")

from dataset import SITSData as supervised_SITSData
from dataset import SSLSITSData as ssl_SITSData
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from models import BYOL, LtaeClassifier
from models.decoder import get_decoder
from models.ltae import LTAE
from utils import standardize
from predetect import evaluation, metrics, post_classification_metrics, train_epoch


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.learner(x)

    def training_step(self, x, _):
        loss = self.forward(x)
        self.log("contrastive_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


def main(args):
    """
    Run self-supervised learning using BYOL framework

    sits 1 & 2: are the test set for the two years
    1. learn the temporal encoder (BYOL), only from pixels with no-change in landcover class prediction
    2. to evaluate the temporal encoder + decoder (supervised learning)
    train_sits: are the training samples for the decoder (Downstream task).
    """
    model_file, preds_18, preds_19, similarity, config_file = get_model_file_path(
        args.model_dir
    )
    config = json.load(open(config_file))

    config["device"] = args.device

    sits_1, train_sits, gdate_1, mean_1, std_1 = get_sits_file_path(
        args.dataset_folder1)  # sits_1 for 2018
    sits_2, _, gdate_2, mean_2, std_2 = get_sits_file_path(
        args.dataset_folder2)  # sits_2 for 2019

    if args.eval_mode == "freeze":
        eval_mode = "freeze"
    elif args.eval_mode == "finetune":
        eval_mode = "finetune"
    else:
        raise ValueError("Invalid eval mode")

    wandb_logger = WandbLogger(
        name=f"tbyol_{eval_mode}", config=args
    )

    transform_1 = transforms.Compose([standardize(mean_1, std_1)])
    transform_2 = transforms.Compose([standardize(mean_2, std_2)])

    ds = ssl_SITSData(
        sits_1,
        sits_2,
        preds_18,
        preds_19,
        similarity,
        date_=gdate_1,
        transform=[transform_1, transform_2],
        args=args,
    )

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size_byol,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    # LTAE model config
    model_config = dict(
        in_channels=config["in_channels"],
        n_head=config["n_head"],
        d_k=config["d_k"],
        n_neurons=config["n_neurons"],
        dropout=config["dropout"],
        d_model=config["d_model"],
        T=config["T"],
        len_max_seq=config["len_max_seq"],
        positions=ds.date_positions if args.positions == "bespoke" else None,
        return_att=False,
    )

    # Temporal encoder
    ltae = LTAE(**model_config)

    if args.start_from_scratch:
        pass
        print("Training from scratch")
    else:
        print("Loading encoder weights from supervised training")
        m_state_dict = torch.load(model_file, map_location=torch.device(config["device"]))
        for k in list(m_state_dict.keys()):
            if k.startswith("temporal_encoder."):
                m_state_dict[k.replace("temporal_encoder.", "")] = m_state_dict[k]
            del m_state_dict[k]
        ltae.load_state_dict(m_state_dict)

    for param in ltae.parameters():
        param.requires_grad = True

    # BYOL config
    byol_config = dict(
        in_channels=config["in_channels"],
        projection_size=args.projection_size,
        projection_hidden_size=args.projection_hidden_size,
        moving_average_decay=args.moving_average_decay,
    )

    model = SelfSupervisedLearner(ltae, learning_rate=args.lr, **byol_config)
    model = model.double()

    model_folder = os.path.join(args.output_dir, f"byol_results")
    # os.makedirs(model_folder, exist_ok=True)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=model_folder,
    #     monitor="contrastive_loss",
    #     mode="min",
    #     filename=f"byol_model_{args.label_mode}",
    #     save_top_k=1,
    # )

    # trainer = pl.Trainer(
    #     accelerator="cpu" if args.device == "cpu" else "gpu",
    #     devices=1 if args.device == "cuda" else None,
    #     max_epochs=args.epochs,
    #     # accumulate_grad_batches=1,
    #     # sync_batchnorm=True,
    #     logger=wandb_logger,
    #     callbacks=[checkpoint_callback],
    # )

    # trainer.fit(model, train_loader)

    print("BYOL completed!")

    #############################
    ##### Downstream task #######
    #############################

    # load save ssl model
    assert os.path.exists(
        os.path.join(model_folder, f"byol_model_{args.label_mode}.ckpt")
    ), "No model file found!"
    
    # load self-supervied model from checkpoint
    checkpoint = torch.load(os.path.join(model_folder, f"byol_model_{args.label_mode}.ckpt"))

    ssl_state_dict = checkpoint["state_dict"]

    # Rename parameter name
    for key in list(ssl_state_dict.keys()):
        if key.startswith("learner.online_encoder.net."):
            ssl_state_dict[
                key.replace("learner.online_encoder.net.", "temporal_encoder.")
            ] = ssl_state_dict[key]
        del ssl_state_dict[key]

    # classifier
    decoder = get_decoder(config["mlp"])
    decoder.apply(weight_init)

    # Update self-supervised model with the classifier
    [
        ssl_state_dict.update({"decoder." + k: v})
        for k, v in decoder.state_dict().items()
    ]

    model_config.update(dict(mlp=config["mlp"]))

    # LtaeClassifier = TemporalEncoder + decoder(classifier)
    model = LtaeClassifier(**model_config).to(torch.device(args.device))

    model.load_state_dict(ssl_state_dict)

    # Evaluation method
    if args.eval_mode == "freeze":
        print("Freeze the encoder")
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False
            
    elif args.eval_mode == "finetune":
        print("Fine-tune the encoder")
        for param in model.parameters():
            param.requires_grad = True

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    model = model.double()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = FocalLoss(config["gamma"])

    train_dt = supervised_SITSData(train_sits, gdate_1, transform=transform_1)
    test_dt18 = supervised_SITSData(sits_1, gdate_1, transform=transform_1)
    test_dt19 = supervised_SITSData(sits_2, gdate_2, transform=transform_2)
    print(f"Train data size: {len(train_dt)}")
    print(f"Test data size: {len(test_dt18)}")
    print(f"Test data size: {len(test_dt19)}")

    train_loader = DataLoader(
        train_dt,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    test_loader18 = DataLoader(
        test_dt18,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    test_loader19 = DataLoader(
        test_dt19,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = train_epoch(model, train_loader, optimizer, criterion, config)
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"Epoch {epoch}: train loss {train_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss})

    torch.save(
        model.state_dict(),
        os.path.join(model_folder, f"dsmodel_{args.label_mode}_{args.eval_mode}.pth"),
    )

    # evaluate on 2018
    _, y_true_18, y_preds_18, proba_18 = evaluation(
        model, test_loader18, criterion, config
    )
    test_metrics_18 = metrics(y_true_18, y_preds_18)
    test_metrics_18 = metrics(y_true_18, y_preds_18)
    test_metrics_msg = " ".join([f"{k}={v:.2f}" for k, v in test_metrics_18.items()])
    print(f"Test18: {test_metrics_msg}")
    test_metrics_18 = {f"test_{k}_2018": v for k, v in test_metrics_18.items()}

    log_folder = os.path.join(model_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)
    
    # Name of landcover classes in the ROI
    label = [
        "Dense built-up area",
        "Diffuse built-up area",
        "Industrial and commercial areas",
        "Roads",
        "Oilseeds (Rapeseed)",
        "Straw cereals (Wheat, Triticale, Barley)",
        "Protein crops (Beans / Peas)",
        "Soy",
        "Sunflower",
        "Corn",
        "Tubers/roots",
        "Grasslands",
        "Orchards and fruit growing",
        "Vineyards",
        "Hardwood forest",
        "Softwood forest",
        "Natural grasslands and pastures",
        "Woody moorlands",
        "Water",
    ]

    # save the classification report
    classif_report = sklearn.metrics.classification_report(
        y_true_18, y_preds_18, target_names=label
    )
    print(
        classif_report,
        file=open(os.path.join(log_folder, "classification_report_18.txt"), "w"),
    )

    # evaluate on 2019
    _, y_true_19, y_preds_19, proba_19 = evaluation(
        model, test_loader19, criterion, config
    )
    test_metrics_19 = metrics(y_true_19, y_preds_19)
    test_metrics_msg = " ".join([f"{k}={v:.2f}" for k, v in test_metrics_19.items()])
    print(f"Test19: {test_metrics_msg}")
    test_metrics_19 = {f"test_{k}_2019": v for k, v in test_metrics_19.items()}

    # save the classification report
    classif_report = sklearn.metrics.classification_report(
        y_true_19, y_preds_19, target_names=label
    )
    print(
        classif_report,
        file=open(os.path.join(log_folder, "classification_report_19.txt"), "w"),
    )

    wandb.log({**test_metrics_18, **test_metrics_19})

    np.save(os.path.join(log_folder, "y_preds_18.npy"), y_true_18)
    np.save(os.path.join(log_folder, "y_preds_19.npy"), y_preds_18)
    np.save(os.path.join(log_folder, "proba_18.npy"), proba_18)
    np.save(os.path.join(log_folder, "proba_19.npy"), proba_19)

    print("Post classification!!")
    # Post classification
    gt_binary = np.where(y_true_18 == y_true_19, 0, 1)
    pred_binary = np.where(y_preds_18 == y_preds_19, 0, 1)
    post_classification_metrics(gt_binary, pred_binary, title="pairwise")

    # similarity using euclidean distance
    proba_18 = torch.from_numpy(proba_18)
    proba_19 = torch.from_numpy(proba_19)
    proba_18 = torch.nn.functional.softmax(proba_18, dim=1)
    proba_19 = torch.nn.functional.softmax(proba_19, dim=1)
    similarity = np.linalg.norm(proba_18 - proba_19, axis=1)
    percentile_thresholds = np.percentile(similarity, np.arange(0, 100, 5))

    f1s = list()
    for threshold in percentile_thresholds:
        pred = np.where(similarity > threshold, 1, 0)
        f1s.append(sklearn.metrics.f1_score(gt_binary, pred))

    best_threshold = percentile_thresholds[np.argmax(f1s)]
    pred_binary = np.where(similarity > best_threshold, 1, 0)
    post_classification_metrics(gt_binary, pred_binary, title="similarity")

    fig, ax = plt.subplots()
    ax.hist(similarity, bins=100, label="similarity distribution")
    ax.axvline(
        x=best_threshold,
        color="red",
        label=f"best threshold={best_threshold:.3f}",
        linestyle="--",
    )
    ax.legend()
    wandb.log({"similarity_threshold": wandb.Image(fig)})

    # save config
    with open(os.path.join(log_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def get_model_file_path(path):
    """Get the supervised classification model and predictions"""
    model_file = glob(os.path.join(path, "*.pth"))[0]
    config_file = os.path.join(path, "logs/config.json")
    similarity = os.path.join(path, "logs/binary_similarity.npy")
    preds_18 = os.path.join(path, "logs/y_preds_18.npy")
    preds_19 = os.path.join(path, "logs/y_preds_19.npy")
    return model_file, preds_18, preds_19, similarity, config_file


def get_sits_file_path(path):
    """Get datasets and files needed for standardization"""
    train_sits = os.path.join(path, "train.npz")
    test_sits = os.path.join(path, "test.npz")
    gapfilled_date = os.path.join(path, "date.txt")
    mean = np.loadtxt(os.path.join(path, "mean.txt"))
    std = np.loadtxt(os.path.join(path, "std.txt"))
    return test_sits, train_sits, gapfilled_date, mean, std


if __name__ == "__main__":
    parser = ArgumentParser()

    # paths
    parser.add_argument("--dataset_folder1", type=str, default="data/sits")
    parser.add_argument("--dataset_folder2", type=str, default="data/sits")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_results/modelname",
    )
    parser.add_argument(
        "--eval_mode", type=str, default="freeze", help="freeze or finetune")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="softlabel",
        help="softlabel, hardlabel, or full_pixel",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_results/",
        help="output directory",
    )
    parser.add_argument(
        "--start_from_scratch", action="store_true", help="train BYOL from scratch"
    )

    # training parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")

    # BYOL parameters
    parser.add_argument("--projection_size", type=int, default=128)
    parser.add_argument("--projection_hidden_size", type=int, default=4096)
    parser.add_argument("--moving_average_decay", type=float, default=0.99)
    parser.add_argument("--batch_size_byol", type=int, default=128)

    parser.add_argument(
        "--positions",
        default="bespoke",
        type=str,
        help="Positions to use for the positional encoding (bespoke / order)",
    )
    parser.set_defaults(start_from_scratch=False)
    args = parser.parse_args()
    main(args)
