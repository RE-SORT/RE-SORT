


import torch
from torch import nn


class CompressedInteractionNet(nn.Module):
    def __init__(self, num_fields, cin_hidden_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_hidden_units = cin_hidden_units
        self.fc = nn.Linear(sum(cin_hidden_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_hidden_units):
            in_channels = num_fields * self.cin_hidden_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, feature_emb):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_hidden_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        output = self.fc(torch.cat(pooling_outputs, dim=-1))
        return output
        

