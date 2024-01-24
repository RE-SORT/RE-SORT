

import torch
from torch import nn
from re_sortctr.pytorch.layers import FeatureEmbedding


class LogisticRegression(nn.Module):
    def __init__(self, feature_map, use_bias=True):
        super(LogisticRegression, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False, use_sharing=False)

    def forward(self, X):
        embed_weights = self.embedding_layer(X)
        output = embed_weights.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        return output

