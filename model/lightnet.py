import torch
import torch.nn as nn
import torch.nn.functional as F

class lightnet(nn.Module):
    
    def __init__(self, args, output_channels=40):
        super(lightnet, self).__init__()
        self.args = args

        # Positional Embedding
        self.pe = Positional_Encoding()
        
        # Input Embedding 将点的特征映射到高维空间 每一个点的特征就是他自身的高维向量
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)

        # Self-Attention
        self.transformer = Self_Attention(args)

        # Stack to 1024 channel
        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        # Classification
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        '''
            x: (b, n, 3)
        '''
        batch_size, _, _ = x.size()
        x_position = self.pe(x)
        x = torch.cat([x, x_position], dim=1)
        
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.transformer(x)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Self_Attention(nn.Module):
    """
    Attention Modules: four stacked attention layer
    Note: no Position Embedding
    """

    def __init__(self, args, channels=256):
        super(Self_Attention, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        return x

class SA_Layer(nn.Module):
    """
    Self-Attention Layer
    """

    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k) # bmm计算两个诸如(b,m,n)和(b,n,l)，得到(b,m,l)

        # Scale
        _, d, _ = x_k.size()
        attention = energy / (d ** 0.5)
        # Softmax
        attention = self.softmax(energy)

        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x

class Positional_Encoding(nn.Module):
    def __init__(self):
        super(Positional_Encoding, self).__init__()
        self.softmax = nn.Softmax(dim=3)
        
    def forward(self, x):
        '''
        Input:
            x: (b, n, 3)

        Output:
            x_position: (b, n, 3)
        '''

        # 计算每个点与其他点的坐标差
        b, n, c = x.size()
        x_repeat = torch.repeat_interleave(x, n, dim=1)
        x_repeat = x_repeat.reshape(b, n, n, c)
        x_diff = torch.abs(x.unsqueeze(1) - x_repeat) # 通过广播机制计算点与点的差值
        x_feat = 1 - self.softmax(x_diff) # 计算每个点的差值贡献分布
        x_new = x.unsqueeze(1) * x_feat # 根据差值分布计算每个点的
        x_position, _ = torch.max(x_new, dim=2)

        return x_position
