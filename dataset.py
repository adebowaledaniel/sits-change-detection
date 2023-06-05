import datetime

import numpy as np
import torch
from torch.utils import data

n_channel = 10 #

class SITSData(data.Dataset):
    def __init__(self, sits, gfdate_path, transform=None):
        """
        Args:
            sits (string): Path to the npz file with SITS.
            gfdate_path (string): Path satetllite image aquistion date
        """
        self.sits = sits
        self.transform = transform
        self.gfdate = gfdate_path

        with np.load(self.sits) as f:
            self.X_ = f["X"]
            self.y_ = f["y"]

        self.date_positions = date_positions(self.gfdate)

    def __len__(self):
        return len(self.y_)

    def __getitem__(self, idx):
        self.X = self.X_[idx]
        self.y = self.y_[idx]

        self.X = np.array(self.X, dtype="float16")
        self.y = np.array(self.y, dtype=int)
        self.X = self.X.reshape(int(self.X.shape[0] / n_channel), n_channel)

        # transform
        if self.transform:
            self.X = self.transform(self.X)

        torch_x = torch.from_numpy(self.X)
        torch_y = torch.from_numpy(self.y)
        return torch_x, torch_y


class SSLSITSData(data.Dataset):
    def __init__(self, sits1, sits2, y1, y2, similarity, date_, transform, args):
        """
        Args:
            sits (path): .npy files for both years
            y (path): predictions for both years
            similarity (path): prediction binary change  
        return:
            X1, X2
        """

        self.sits1 = sits1
        self.sits2 = sits2
        self.date_ = date_
        self.transform_1 = transform[0]
        self.transform_2 = transform[1]

        if args.label_mode == "full_pixel":
            print("Loading full pixel data")
            self.X1_ = np.load(self.sits1)["X"]
            self.X2_ = np.load(self.sits2)["X"]
        elif args.label_mode == "softlabel":
            print("Loading similarity data")
            similarity = np.load(similarity)
            mask = similarity == 0

            self.X1_ = np.load(self.sits1)["X"]
            self.X1_ = self.X1_[mask]
            self.X2_ = np.load(self.sits2)["X"]
            self.X2_ = self.X2_[mask]
        elif args.label_mode == "hardlabel":
            print("Loading only no changed pixel data")
            y1 = np.load(y1)  
            y2 = np.load(y2)  
            mask = y1 == y2

            self.X1_ = np.load(self.sits1)["X"]  
            self.X1_ = self.X1_[mask]
            self.X2_ = np.load(self.sits2)["X"]
            self.X2_ = self.X2_[mask]
        else:
            raise ValueError("Invalid mode")

        self.date_positions = date_positions(date_)

    def __len__(self):
        assert len(self.X1_) == len(self.X2_), "X1 and X2 must be of same length"
        return len(self.X1_)

    def __getitem__(self, idx):
        self.X1 = self.X1_[idx]
        self.X2 = self.X2_[idx]

        self.X1 = np.array(self.X1, dtype="float16")
        self.X2 = np.array(self.X2, dtype="float16")
        self.X1 = self.X1.reshape(int(self.X1.shape[0] / n_channel), n_channel)
        self.X2 = self.X2.reshape(int(self.X2.shape[0] / n_channel), n_channel)

        if self.transform_1 and self.transform_2 is not None:
            self.X1 = self.transform_1(self.X1)
            self.X2 = self.transform_2(self.X2)

        torch_x1 = torch.from_numpy(self.X1)
        torch_x2 = torch.from_numpy(self.X2)
        return torch_x1, torch_x2


def date_positions(gfdate_path):
    """Return DOY"""
    with open(gfdate_path, "r") as f:
        date_list = f.readlines()
    date_list = [x.strip() for x in date_list]
    date_list = [
        datetime.datetime.strptime(x, "%Y%m%d").timetuple().tm_yday for x in date_list
    ]
    final_date_llist = [x for x in date_list]
    return final_date_llist