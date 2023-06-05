import argparse
import json
import os
import pprint
import sys
from glob import glob

import numpy as np
import seaborn as sns
import sklearn.metrics
import wandb

sys.path.append("..")

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import SITSData
from learning.focal_loss import FocalLoss
from learning.metrics import mIou
from learning.weight_init import weight_init
from models import LtaeClassifier, TempCNN
from utils import standardize

warnings.filterwarnings("ignore")

def train(config):
    """
    Run supervised classification
    """
    print("get loaders")
    dataloaders, date_position = get_loader(config)

    model = get_model(config, date_position)
    model.modelname += f"_epochs={config['epochs']}"

    wandb.init(name=model.modelname, config=config)

    for (
        traindataloader,
        valdataloader,
        testdataloader_18,
        testdataloader_19,
    ) in dataloaders:
        print(
            f"Train: {len(traindataloader)}, Val:{len(valdataloader)}, Test18: {len(testdataloader_18)}, Test19: {len(testdataloader_19)}"
        )
        criterion = get_criterion(config)
        optimizer = get_optimizer(config, model)

        best_mIoU = 0

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

        vallog = list()
        trainlog = list()
        for epoch in range(1, config["epochs"] + 1):
            # Train
            train_loss, yt_train, yp_train = train_epoch(
                model, traindataloader, optimizer, criterion, config
            )
            train_metrics = metrics(yt_train, yp_train)
            train_loss = train_loss.cpu().detach().numpy()[0]

            # Validation
            val_loss, yt_val, yp_val, _ = evaluation(
                model, valdataloader, criterion, config
            )

            val_metrics = metrics(yt_val, yp_val)
            val_metrics_msg = " ".join([f"{k}={v:.2f}" for k, v in val_metrics.items()])
            val_loss = val_loss.cpu().detach().numpy()[0]

            del yt_train, yp_train, yt_val, yp_val

            print(
                f"Epoch {epoch}/{config['epochs']}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f}, {val_metrics_msg}"
            )

            train_metrics["loss"] = train_loss
            val_metrics["loss"] = val_loss

            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            train_metrics["epoch"] = epoch
            val_metrics["epoch"] = epoch

            wandb.log(train_metrics)
            wandb.log(val_metrics)

            vallog.append(val_metrics)
            trainlog.append(train_metrics)

            # save model and logs
            model_folder = os.path.join(config["output_folder"], model.modelname)
            os.makedirs(model_folder, exist_ok=True)
            log_folder = os.path.join(model_folder, "logs")
            os.makedirs(log_folder, exist_ok=True)

            save_log(vallog, "val", log_folder)
            save_log(trainlog, "train", log_folder)

            # Save best model
            if val_metrics["val_miou"] >= best_mIoU:
                best_mIoU = val_metrics["val_miou"]
                model_state = model.state_dict()
                torch.save(
                    model_state, os.path.join(model_folder, model.modelname + ".pth")
                )

        # # Test on best model saved
        print("Testing on best model")
        model.load_state_dict(
            torch.load(os.path.join(model_folder, model.modelname + ".pth"))
        )

        # Test on 2018
        _, y_true_18, y_preds_18, proba_18 = evaluation(
            model, testdataloader_18, criterion, config
        )
        test_metrics_18 = metrics(y_true_18, y_preds_18)

        # save the classification report
        classif_report = sklearn.metrics.classification_report(
            y_true_18, y_preds_18, target_names=label
        )
        print(
            classif_report,
            file=open(os.path.join(log_folder, "classification_report_18.txt"), "w"),)

        test_metrics_18 = {f"test_{k}_2018": v for k, v in test_metrics_18.items()}

        # Test on 2019
        _, y_true_19, y_preds_19, proba_19 = evaluation(
            model, testdataloader_19, criterion, config
        )
        test_metrics_19 = metrics(y_true_19, y_preds_19)

        # save the classification report
        classif_report = sklearn.metrics.classification_report(
            y_true_19, y_preds_19, target_names=label)
        
        print(
            classif_report,
            file=open(os.path.join(log_folder, "classification_report_19.txt"), "w"),
        )

        test_metrics_19 = {f"test_{k}_2019": v for k, v in test_metrics_19.items()}

        wandb.log({**test_metrics_18, **test_metrics_19})

        print(f"saving y_preds in {log_folder}")
        np.save(os.path.join(log_folder, "y_preds_18.npy"), y_preds_18)
        np.save(os.path.join(log_folder, "y_preds_19.npy"), y_preds_19)
        np.save(os.path.join(log_folder, "proba_18.npy"), proba_18)
        np.save(os.path.join(log_folder, "proba_19.npy"), proba_19)

        print("Post classification!!")
        # post classification
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

        # save binary similarity
        np.save(os.path.join(log_folder, "binary_similarity.npy"), pred_binary)

        # save config
        with open(os.path.join(log_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

def train_epoch(model, loader, optimizer, criterion, config):
    model.train()
    losses = list()
    y_true = list()
    y_pred = list()
    with tqdm(enumerate(loader), total=len(loader)) as iterator:
        for idx, (x, y) in iterator:
            optimizer.zero_grad()
            out = model.forward(x.to(torch.device(config["device"])))
            loss = criterion(out, y.to(torch.device(config["device"])))
            loss.backward()
            optimizer.step()

            pred = out.detach()
            y_pd = pred.argmax(dim=1).cpu().numpy()
            y_true.append(y.cpu().numpy())
            y_pred.append(y_pd)

            iterator.set_description(
                f"Train Step [{idx + 1}/{len(loader)}], Loss: {loss:.2f}"
            )
            losses.append(loss)
        return torch.stack(losses), np.concatenate(y_true), np.concatenate(y_pred)


def evaluation(model, loader, criterion, config):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true = list()
        y_pred = list()
        probabilities = list()
        with tqdm(enumerate(loader), total=len(loader)) as iterator:
            for idx, (x, y) in iterator:
                out = model.forward(x.to(torch.device(config["device"])))
                loss = criterion(out, y.to(torch.device(config["device"])))

                iterator.set_description(
                    f"Eval [{idx + 1}/{len(loader)}], Loss: {loss:.2f}"
                )
                losses.append(loss)
                y_true.append(y.cpu().numpy())
                y_pd = out.argmax(dim=1).cpu().numpy()
                y_pred.append(y_pd)
                probabilities.append(out.cpu())
    return (
        torch.stack(losses),
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(probabilities),
    )

def save_log(log, mode, log_folder):
    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(log_folder, f"{mode}_log.csv"))

def post_classification_metrics(gt_binary, otsu_binary, title=""):
    f1 = sklearn.metrics.f1_score(gt_binary, otsu_binary)
    kappa = sklearn.metrics.cohen_kappa_score(gt_binary, otsu_binary)
    wandb.log({f"{title} pc-f1": f1})
    wandb.log({f"{title} pc-kappa": kappa})
    cmatrix = sklearn.metrics.confusion_matrix(gt_binary, otsu_binary)
    cmatrixper = cmatrix.astype("float") / np.sum(cmatrix)
    label = ["No change", "Change"]
    sns_plot = sns.heatmap(
        (cmatrixper * 100),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label,
        yticklabels=label,
        cbar=False,
        annot_kws={"size": 15},
    )
    sns_plot.set(xlabel="Predicted", ylabel="Ground truth")

    wandb.log({f"{title} Total error": (cmatrixper[0, 1] + cmatrixper[1, 0]) * 100})
    wandb.log({f"{title} False alarm": cmatrixper[0, 1] * 100})
    wandb.log({f"{title} Missed detection": cmatrixper[1, 0] * 100})
    wandb.log({f"{title} Correct detection": cmatrixper[1, 1] * 100})
    wandb.log({f"{title} Correct non-detection": cmatrixper[0, 0] * 100})
    wandb.log({f"{title} confusion_matrix": wandb.Image(sns_plot)})

    del cmatrix
    del cmatrixper


def metrics(y_true, y_pred):
    # source: https://github.com/dl4sits/BreizhCrops/blob/master/examples/train.py

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    miou = mIou(y_true, y_pred, n_classes=19)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(
        y_true, y_pred, average="weighted"
    )

    return dict(
        accuracy=accuracy,
        miou=miou,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


def get_optimizer(config, model):
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    else:
        raise ValueError("Optimizer not supported")
    return optimizer


def get_criterion(config):
    if config["loss"] == "focal":
        criterion = FocalLoss(config["gamma"])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def get_model(config, date_positions=None):
    if config["model"] == "tempcnn":
        model = TempCNN(
            input_dim=config["in_channels"],
            num_classes=config["num_classes"],
            sequencelength=config["len_max_seq"],
        )
        model = model.to(torch.device(config["device"]))
        config["N_parameters"] = model.param_ratio()

    elif config["model"] == "ltae":
        model_config = dict(
            in_channels=config["in_channels"],
            n_head=config["n_head"],
            d_k=config["d_k"],
            n_neurons=config["n_neurons"],
            dropout=config["dropout"],
            d_model=config["d_model"],
            mlp=config["mlp"],
            T=config["T"],
            len_max_seq=config["len_max_seq"],
            positions=date_positions if config["positions"] == "bespoke" else None,
        )

        model = LtaeClassifier(**model_config).to(torch.device(config["device"]))
        config["N_parameters"] = model.param_ratio()
        model.apply(weight_init)
        model = model.double()

    else:
        raise ValueError("Invalid model name")
    return model


def get_loader(config):
    # eo mean & std data standardization transform
    dataset_folder1 = os.path.join(config["dataset_folder1"])
    dataset_folder2 = os.path.join(config["dataset_folder2"])
    
    load_paths = get_paths(dataset_folder1, dataset_folder2)
    transform_18 = transforms.Compose([standardize(load_paths['mean_18'], load_paths['std_18'])])
    transform_19 = transforms.Compose([standardize(load_paths['mean_19'], load_paths['std_19'])])

    print("Loading data")
    train_dt = SITSData(load_paths['train_sits_data'], load_paths['doy18'], transform=transform_18)
    print("Train data loaded")
    val_dt = SITSData(load_paths['val_sits_data'], load_paths['doy18'], transform=transform_18)
    print("Val data loaded")
    test_dt_18 = SITSData(load_paths['test_2018_sits'], load_paths['doy18'], transform=transform_18)
    test_dt_19 = SITSData(load_paths['test_2019_sits'], load_paths['doy_19'], transform=transform_19)
    print("Data loaded")

    loader_seq = []
    train_loader = data.DataLoader(
        train_dt,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_dt,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    test_loader_18 = data.DataLoader(
        test_dt_18,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
    )

    test_dt_19 = data.DataLoader(
        test_dt_19,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    print("Dataloader created")
    loader_seq.append((train_loader, val_loader, test_loader_18, test_dt_19))
    return loader_seq, test_dt_18.date_positions

def get_paths(dataset_folder1, dataset_folder2):
        train_sits_data = dataset_folder1 + f"/train.npz"
        val_sits_data = dataset_folder1 + f"/val.npz"

        test_2018_sits = dataset_folder1 + "/test.npz"
        test_2019_sits = os.path.join(dataset_folder2, "test.npz")

        doy18 = dataset_folder1 + "/date.txt"
        doy_19 = dataset_folder2 + "/date.txt"

        mean_18 = np.loadtxt(dataset_folder1 + "/mean.txt")
        std_18 = np.loadtxt(dataset_folder1 + "/std.txt")

        mean_19 = np.loadtxt(dataset_folder2 + "/mean.txt")
        std_19 = np.loadtxt(dataset_folder2 + "/std.txt")

        return {
            "train_sits_data": train_sits_data,
            "val_sits_data": val_sits_data,
            "test_2018_sits": test_2018_sits,
            "test_2019_sits": test_2019_sits,
            "mean_18": mean_18,
            "std_18": std_18,
            "mean_19": mean_19,
            "std_19": std_19,
            "doy18": doy18,
            "doy_19": doy_19,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument(
        "--dataset_folder1",
        default="data/sits",
        type=str,
        help="Path to the data folder.",
    )
    parser.add_argument(
        "--dataset_folder2",
        default="data/sits",
        type=str,
        help="Path to the data folder.",
    )
    parser.add_argument(
        "--output_folder",
        default="model_results/",
        type=str,
        help="Path to the model folder.",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--model",
        default="ltae",
        type=str,
        help="Model to use for training.",
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--lr",
        default=0.00001,
        type=float,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--gamma", default=1, type=float, help="Gamma parameter of the focal loss"
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--loss",
        default="focal",
        type=str,
        help="Loss function to use for training.",
    )

    # Data parameters
    parser.add_argument(
        "--in_channels",
        default=10,
        type=int,
        help="Number of channels of the input embeddings",
    )
    parser.add_argument(
        "--num_classes",
        default=19,
        type=int,
        help="Number of classes for the classification task.",
    )
    parser.add_argument(
        "--len_max_seq",
        default=53,
        type=int,
        help="Maximum sequence length for positional encoding",
    )

    # LTAE parameters
    parser.add_argument(
        "--n_head", default=16, type=int, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_k", default=8, type=int, help="Dimension of the key and query vectors"
    )
    parser.add_argument(
        "--n_neurons",
        default="[128, 64]",
        type=str,
        help="Number of neurons in the layers of n_neurons",
    )
    parser.add_argument(
        "--T", default=1000, type=int, help="Maximum period for the positional encoding"
    )
    parser.add_argument(
        "--positions",
        default="bespoke",
        type=str,
        help="Positions to use for the positional encoding (bespoke / order)",
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="Dropout probability"
    )
    parser.add_argument(
        "--d_model",
        default=128,
        type=int,
        help="size of the embeddings (E), if input vectors are of a different size, a linear layer is used to project them to a d_model-dimensional space",
    )
    parser.add_argument(
        "--mlp",
        default="[64, 32, 19]",
        type=str,
        help="Number of neurons in the layers of MLP (Decoder)",
    )

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if "mlp" in k or k == "n_neurons":
            v = v.replace("[", "")
            v = v.replace("]", "")
            config[k] = list(map(int, v.split(",")))

    pprint.pprint(config)

    train(config)
