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
