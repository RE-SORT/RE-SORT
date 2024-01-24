


import torch
from torch import nn
from itertools import combinations


class HolographicInteraction(nn.Module):
    def __init__(self, num_fields, interaction_type="circular_convolution"):
        super(HolographicInteraction, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == "circular_correlation":
            self.conj_sign =  nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)
        self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)

    def forward(self, feature_emb):
        emb1 =  torch.index_select(feature_emb, 1, self.triu_index[0])
        emb2 = torch.index_select(feature_emb, 1, self.triu_index[1])
        if self.interaction_type == "hadamard_product":
            interact_tensor = emb1 * emb2
        elif self.interaction_type == "circular_convolution":
            fft1 = torch.view_as_real(torch.fft.fft(emb1))
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        elif self.interaction_type == "circular_correlation":
            fft1_emb = torch.view_as_real(torch.fft.fft(emb1))
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        else:
            raise ValueError("interaction_type={} not supported.".format(self.interaction_type))
        return interact_tensor
