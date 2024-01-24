


import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3, excitation_activation="ReLU"):
        super(SqueezeExcitation, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        excitation = [nn.Linear(num_fields, reduced_size, bias=False),
                      nn.ReLU(),
                      nn.Linear(reduced_size, num_fields, bias=False)]
        if excitation_activation.lower() == "relu":
            excitation.append(nn.ReLU())
        elif excitation_activation.lower() == "sigmoid":
            excitation.append(nn.Sigmoid())
        else:
            raise NotImplementedError
        self.excitation = nn.Sequential(*excitation)

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V
        