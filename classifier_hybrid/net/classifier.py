import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


class Classifier(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, in_features, num_classes, graph_args,
                 temporal_kernel_size=75, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn1 = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 32, kernel_size, 1, residual=False, **kwargs),
            st_gcn(32, 64, kernel_size, 2, **kwargs),
            st_gcn(64, 64, kernel_size, 2, **kwargs)
        ))
        # fcn for combining and predicting
        self.data_bn2 = nn.BatchNorm1d(64+in_features)
        self.combined_networks = nn.ModuleList((
            nn.Conv2d(64+in_features, 96, kernel_size=1),
            nn.Conv2d(96, num_classes, kernel_size=1)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


    def forward(self, x_aff, x_gait):

        # data normalization
        N, C, T, V, M = x_gait.size()
        x = x_gait.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn1(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # forward combined and prediction
        x_aff = x_aff.unsqueeze(2).unsqueeze(2)
        x = torch.cat((x, x_aff), dim=1)
        x = self.data_bn2(x.squeeze()).unsqueeze(2).unsqueeze(2)
        for net in self.combined_networks:
            x = net(x)

        x = x.view(x.size(0), -1)

        return x
    # def forward(self, x_gait, x_aff):
    # # Chuẩn hóa dữ liệu
    #     N, C, T, V, M = x_gait.size()
    #     x = x_gait.permute(0, 4, 3, 1, 2).contiguous()
    #     x = x.view(N * M, V * C, T)
    #     x = self.data_bn1(x)
    #     x = x.view(N, M, V, C, T)
    #     x = x.permute(0, 1, 3, 4, 2).contiguous()
    #     x = x.view(N * M, C, T, V)

    #     # ST-GCN layers
    #     for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
    #         x, _ = gcn(x, self.A * importance)

    #     # Global Average Pooling
    #     x = F.avg_pool2d(x, x.size()[2:])
    #     x = x.view(N, M, -1, 1, 1).mean(dim=1)  # Kích thước (N, 64, 1, 1)
        
    #     # Kết hợp với affective features
    #     x_aff = x_aff.unsqueeze(2).unsqueeze(2)  # Thêm chiều để phù hợp (N, in_features, 1, 1)
    #     x = torch.cat((x, x_aff), dim=1)  # Ghép chiều kênh: (N, 64 + in_features, 1, 1)

    #     # Conv2D (1x1) để kết hợp affective features
    #     x = self.affective_conv(x)

    #     # Flatten và đưa vào Fully Connected
    #     x = x.view(x.size(0), -1)  # Kích thước (N, 64 + in_features)
    #     x = self.fc(x)  # (N, num_classes)
    #     x = F.softmax(x, dim=1)

    #     return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
