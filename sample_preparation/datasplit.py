import argparse
import os
import random
import sys

import geopandas as gpd
import numpy as np
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import loaddata, read_ids


def dataset_split_func(sits, grid_path, yraster, idraster, pers, seed, split, outdir):
    """
    Split the dataset into train, val and test sets.
    Prequsite: grid_split.py
    Args:
        sits: Path to the Sentinel-2 datacube
        grid_path: Path to the grid shapefile
        yraster: Path to the rasterized sample label
        idraster: Path to the rasterized  sample polygon id
        split: Split type. Either trainval or test
        seed: Random seed
    """
    train_ids, val_ids, test_ids = read_ids()
    train_ids_val = train_ids + val_ids

    if split == "trainval":
        ids = train_ids_val
    elif split == "test":
        ids = test_ids

    print("loading data...")
    gdf = gpd.read_file(grid_path)

    X = []
    Y = []
    polygon_ids = []

    for i in tqdm(ids):
        bounds = gdf[gdf["FID"] + 1 == i].total_bounds
        x, y, ids = loaddata(sits, yraster, idraster, bounds)
        X.append(x.transpose(1, 2, 0).reshape(-1, x.shape[0]))
        Y.append(y.flatten())
        polygon_ids.append(ids.flatten())

    print("Concatenating data...")

    mask = np.concatenate(Y) != 0
    X = np.concatenate(X)[mask]
    y = np.concatenate(Y)[mask]
    polygon_ids = np.concatenate(polygon_ids)[mask]
    y = np.unique(y, return_inverse=True)[1]  # reassigning label [1,23] to [0,18]

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

    print("Splitting data...")

    # Only for test data
    if split == "test":
        print("Saving data...")
        np.savez_compressed(os.path.join(outdir, f"test_per100.npz"), X=X, y=y)
        assert os.path.exists(
            os.path.join(outdir, f"test_per100.npz")
        ), "Test data not saved"
        unique, counts = np.unique(y, return_counts=True)
        unique = [label[i] for i in unique]
        percent = np.round(counts / np.sum(counts) * 100, 2)

        wandb.init(
            project="percentage_distribution_per_label",
            name=f"{split}_100",
        )
        wandb.log(
            {
                "Distribution per label (Test)": wandb.Table(
                    data=np.array([unique, counts, percent]).T,
                    columns=["Test Class", "Test Count", "percent"],
                )
            }
        )
        wandb.finish()

    else:
        pers = [int(i) for i in pers]  # example [10,25,50,100]

        sgkf_all = StratifiedGroupKFold(n_splits=100, shuffle=True, random_state=seed)
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)

        unique_y = np.unique(y)
        init_train_ind = []
        init_val_ind = []
        for uy in unique_y:
            y_indices = np.where(y == uy)[0]
            unique_pids = np.unique(polygon_ids[y_indices])
            random.shuffle(unique_pids)
            init_train_ind.extend(np.where(polygon_ids == unique_pids[0])[0])
            init_val_ind.extend(np.where(polygon_ids == unique_pids[1])[0])

        init_X_train, init_X_val = X[init_train_ind], X[init_val_ind]
        init_y_train, init_y_val = y[init_train_ind], y[init_val_ind]
        init_pids_train, init_pids_val = (
            polygon_ids[init_train_ind],
            polygon_ids[init_val_ind],
        )

        X = np.delete(X, init_train_ind + init_val_ind, axis=0)
        y = np.delete(y, init_train_ind + init_val_ind)
        polygon_ids = np.delete(polygon_ids, init_train_ind + init_val_ind)

        for per in tqdm(pers):
            wandb.init(
                project="percentage_distribution_per_label",
                name=f"{split}_{per}",
            )
            assert (
                split == "trainval"
            ), "Only trainval split is supported for partial data"

            if per != 100:
                nfold = per
                cmpt = 0
                per_index = []
                for i, (_, _index) in enumerate(sgkf_all.split(X, y, polygon_ids)):
                    per_index.extend(_index)
                    cmpt += 1
                    if cmpt == nfold:
                        break
                X_ = X[per_index]
                y_ = y[per_index]
                polygon_ids_ = polygon_ids[per_index]
                for train_index, val_index in sgkf.split(X_, y_, polygon_ids_):
                    X_train, X_val = X_[train_index], X_[val_index]
                    y_train, y_val = y_[train_index], y_[val_index]
                    pids_train, pids_val = (
                        polygon_ids_[train_index],
                        polygon_ids_[val_index],
                    )
                    break
                X_train, X_val = np.concatenate(
                    (init_X_train, X_train)
                ), np.concatenate((init_X_val, X_val))
                y_train, y_val = np.concatenate(
                    (init_y_train, y_train)
                ), np.concatenate((init_y_val, y_val))
                pids_train, pids_val = np.concatenate(
                    (init_pids_train, pids_train)
                ), np.concatenate((init_pids_val, pids_val))

            else:
                for train_index, val_index in sgkf.split(X, y, polygon_ids):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    pids_train, pids_val = (
                        polygon_ids[train_index],
                        polygon_ids[val_index],
                    )
                    break

            print("Saving data...")
            np.savez_compressed(
                os.path.join(outdir, f"train_per{per}.npz"),
                X=X_train,
                y=y_train,
                pids=pids_train,
            )
            np.savez_compressed(
                os.path.join(outdir, f"val_per{per}.npz"),
                X=X_val,
                y=y_val,
                pids=pids_val,
            )

            # log class distribution
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            train_unique = [label[i] for i in train_unique]
            train_percent = np.round(train_counts / np.sum(train_counts) * 100, 2)
            val_unique, val_counts = np.unique(y_val, return_counts=True)
            val_unique = [label[i] for i in val_unique]
            val_percent = np.round(val_counts / np.sum(val_counts) * 100, 2)

            wandb.log(
                {
                    "Distribution per label (Train&Val)": wandb.Table(
                        data=np.array(
                            [
                                train_unique,
                                train_counts,
                                train_percent,
                                val_unique,
                                val_counts,
                                val_percent,
                            ]
                        ).T,
                        columns=[
                            "Train Class",
                            "Train Count",
                            "Train Percent",
                            "Val Class",
                            "Val Count",
                            "Val Percent",
                        ],
                    )
                }
            )
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sits", type=str, help="path to the sits npz file",)
    parser.add_argument(
        "--grid_path", type=str,help="path to the grid shapefile",)
    parser.add_argument("--yraster",type=str,help="Path to the rasterized sample label")
    parser.add_argument(
        "--idraster", type=str, help="path to the raster file containing the ids of the polygons",)
    parser.add_argument(
        "--split", type=str, default="trainval", help="split to be used for training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the random number generator"
    )
    parser.add_argument(
        "--pers", type=str, help="percentage of the data to be used for training"
    )
    parser.add_argument(
        "--outdir", type=str, help="output directory")

    config = parser.parse_args()
    for k, v in config.__dict__.items():
        if k == "pers":
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__dict__[k] = v.split(",")
    dataset_split_func(
        config.sits,
        config.grid_path,
        config.yraster,
        config.idraster,
        config.seed,
        config.pers,
        config.split,
        config.outdir,
    )
