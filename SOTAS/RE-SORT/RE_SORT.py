import torch
from torch import nn
from FCE import BaseModel
from re_sortctr.pytorch.layers import FeatureEmbedding, MLP_Block
from .xpos_relative_position import XPOS
import math
class RE_SORT(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="RE_SORT",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 MSR_layers=2,
                 MSR_dim=8,
                 net_dropout=0,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 use_fs=True,
                 group_norm=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(RE_SORT, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512),MSR_layers))).detach().cpu().tolist()
        self.self_MSR1 = nn.Sequential(
            *[MultiHeadSelfMSR(feature_dim if i == 0 else MSR_dim,
                                     MSR_dim=MSR_dim,
                                     num_heads=num_heads,
                                     dropout_rate=net_dropout,
                                     use_residual=use_residual,
                                     use_scale=use_scale,
                                     group_norm=group_norm,#) \
                                     gamma=self.gammas[i]) \
              for i in range(MSR_layers)])
        self.self_MSR2 = nn.Sequential(
            *[MultiHeadSelfMSR(feature_dim if i == 0 else MSR_dim,
                                     MSR_dim=MSR_dim,
                                     num_heads=num_heads,
                                     dropout_rate=net_dropout,
                                     use_residual=use_residual,
                                     use_scale=use_scale,
                                     group_norm=group_norm,#) \
                                     gamma=self.gammas[i]) \
              for i in range(MSR_layers)])
        self.fc = nn.Linear(feature_map.num_fields * MSR_dim, 1)
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_map, 
                                              feature_dim, 
                                              embedding_dim, 
                                              fs_hidden_units, 
                                              fs1_context,
                                              fs2_context)
        self.fusion_module = InteractionAggregation(MSR_dim, 
                                                    MSR_dim, 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.self_MSR1(feat1), self.self_MSR2(feat2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FeatureSelection(nn.Module):
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=[], fs2_context=[]):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs1_context)
        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs2_context)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
class RetDotProductMSR(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(RetDotProductMSR, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, D ,scale=None, mask=None):
        # mask: 0 for masked positions
        #print("type(Q)=",type(Q))
        #print("type(D)=",type(D))
        scores = torch.matmul(Q, K.transpose(-1, -2)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        #MSR = scores.softmax(dim=-1)
        MSR = scores* D.unsqueeze(0).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.dropout is not None:
            MSR = self.dropout(MSR)
        output = torch.matmul(MSR, V)
        return output, MSR





class MultiHeadSelfMSR(nn.Module):
    """ Multi-head MSR module """

    def __init__(self, input_dim, MSR_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, group_norm=True,gamma=0.9):
        super(MultiHeadSelfMSR, self).__init__()
        if MSR_dim is None:
            MSR_dim = input_dim
        assert MSR_dim % num_heads == 0, \
            "MSR_dim={} is not divisible by num_heads={}".format(MSR_dim, num_heads)
        self.head_dim = MSR_dim // num_heads
        self.MSR_dim = MSR_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, MSR_dim, bias=False)
        self.W_k = nn.Linear(input_dim, MSR_dim, bias=False)
        self.W_v = nn.Linear(input_dim, MSR_dim, bias=False)
        self.xpos = XPOS(MSR_dim)
        self.gamma = gamma
        self.swish = lambda x: x * torch.sigmoid(x)
        #self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), num_heads))).detach().cpu().tolist()
        if self.use_residual and input_dim != MSR_dim:
            self.W_res = nn.Linear(input_dim, MSR_dim, bias=False)
        else:
            self.W_res = None
        self.dot_MSR = RetDotProductMSR(dropout_rate)
        self.group_norm_2 = group_norm
        self.group_norm = nn.GroupNorm(num_heads, MSR_dim) if group_norm else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D

    def forward(self, X):
        residual = X

        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        query = self.xpos(query)
        key = self.xpos(key)
        if self.group_norm_2 is not None:
            #print("Done group_norm")
            key = self.group_norm(key.reshape(-1, self.MSR_dim)).reshape(query.shape)


        value = self.W_v(X)

        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        # scaled dot product MSR
        output, MSR = self.dot_MSR(self.swish(query), key, value,D, scale=self.scale)
        #output, MSR = self.dot_MSR(query, key, value,D, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        #if self.group_norm_2 is not None:
         #   output = self.group_norm(output)
        output = output.relu()
        return output
