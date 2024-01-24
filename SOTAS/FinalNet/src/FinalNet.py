

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from re_sortctr.pytorch.models import BaseModel
from re_sortctr.pytorch.layers import FeatureEmbedding, MLP_Block
from re_sortctr.pytorch.torch_utils import get_activation


class FinalNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FinalNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 block_type="2B",
                 batch_norm=True,
                 use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64],
                 block1_hidden_activations=None,
                 block1_dropout=0,
                 block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None,
                 block2_dropout=0,
                 residual_type="concat",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalNet, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        assert block_type in ["1B", "2B"], "block_type={} not supported.".format(block_type)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.use_feature_gating = use_feature_gating
        if use_feature_gating:
            self.feature_gating = FeatureGating(num_fields, gate_residual="concat")
            gate_out_dim = embedding_dim * num_fields * 2
        self.block_type = block_type
        self.block1 = FinalBlock(input_dim=gate_out_dim if use_feature_gating \
                                           else embedding_dim * num_fields,
                                 hidden_units=block1_hidden_units,
                                 hidden_activations=block1_hidden_activations,
                                 dropout_rates=block1_dropout,
                                 batch_norm=batch_norm,
                                 residual_type=residual_type)
        self.fc1 = nn.Linear(block1_hidden_units[-1], 1)
        if block_type == "2B":
            self.block2 = FinalBlock(input_dim=embedding_dim * num_fields,
                                     hidden_units=block2_hidden_units,
                                     hidden_activations=block2_hidden_activations,
                                     dropout_rates=block2_dropout,
                                     batch_norm=batch_norm,
                                     residual_type=residual_type)
            self.fc2 = nn.Linear(block2_hidden_units[-1], 1)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred, y1, y2 = None, None, None
        if self.block_type == "1B":
            y_pred = self.forward1(feature_emb)
        elif self.block_type == "2B":
            y1 = self.forward1(feature_emb)
            y2 = self.forward2(feature_emb)
            y_pred = 0.5 * (y1 + y2)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "y1": y1, "y2": y2}
        return return_dict

    def forward1(self, X):
        if self.use_feature_gating:
            X = self.feature_gating(X)
        block1_out = self.block1(X.flatten(start_dim=1))
        y_pred = self.fc1(block1_out)
        return y_pred

    def forward2(self, X):
        block2_out = self.block2(X.flatten(start_dim=1))
        y_pred = self.fc2(block2_out)
        return y_pred

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.block_type == "2B":
            y1 = self.output_activation(return_dict["y1"])
            y2 = self.output_activation(return_dict["y2"])
            loss1 = self.loss_fn(y1, return_dict["y_pred"].detach(), reduction='mean')
            loss2 = self.loss_fn(y2, return_dict["y_pred"].detach(), reduction='mean')
            loss = loss + loss1 + loss2
        return loss


class FeatureGating(nn.Module):
    def __init__(self, num_fields, gate_residual="concat"):
        super(FeatureGating, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        assert gate_residual in ["concat", "sum"]
        self.gate_residual = gate_residual

    def reset_custom_params(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        if self.gate_residual == "concat":
            out = torch.cat([feature_emb, feature_emb * gates], dim=1) # b x 2f x d
        else:
            out = feature_emb + feature_emb * gates
        return out


class FinalBlock(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None, 
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        # Factorized Interaction Block: Replacement of MLP block
        super(FinalBlock, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FactorizedQuadraticInteraction(hidden_units[idx],
                                                             hidden_units[idx + 1],
                                                             residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class FactorizedInteraction(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        """ FactorizedInteraction layer is an improvement of nn.Linear to capture quadratic 
            interactions between features.
            Setting `residual_type="concat"` keeps the same number of parameters as nn.Linear
            while `residual_type="sum"` doubles the number of parameters.
        """
        super(FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            h = torch.cat([h2, h1 * h2], dim=-1)
        elif self.residual_type == "sum":
            h = h2 + h1 * h2
        return h
