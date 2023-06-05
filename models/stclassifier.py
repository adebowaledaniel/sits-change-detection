import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import get_decoder
from models.ltae import LTAE

__all__ = ["LtaeClassifier"]


class LtaeClassifier(nn.Module):
    """
    LTAE + MLP Classifier"""

    def __init__(
        self,
        in_channels=10,
        n_head=16,
        d_k=8,
        n_neurons=[256, 128],
        dropout=0.2,
        d_model=256,
        mlp=[128, 64, 32, 19],
        T=1000,
        len_max_seq=53,
        positions=None,
        return_att=False,
    ):
        super(LtaeClassifier, self).__init__()

        self.modelname = f"Ltae_d-model={d_model}_mlp={''.join(map(str,mlp))}"

        self.temporal_encoder = LTAE(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            n_neurons=n_neurons,
            dropout=dropout,
            d_model=d_model,
            T=T,
            len_max_seq=len_max_seq,
            positions=positions,
            return_att=return_att,
        )

        self.decoder = get_decoder(mlp)
        self.return_att = return_att

    def forward(self, input):
        """ """
        if self.return_att:
            out, att = self.temporal_encoder(input)
            out = self.decoder(out)
            return out, att
        else:
            out = self.temporal_encoder(input)
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print("TOTAL TRAINABLE PARAMETERS : {}".format(total))
        print(
            "RATIOS: Temporal {:5.1f}% , Classifier {:5.1f}%".format(
                t / total * 100, c / total * 100
            )
        )
        return total


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
