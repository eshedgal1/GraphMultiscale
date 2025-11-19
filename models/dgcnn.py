import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn import global_max_pool
from spatial_transformer_net import *


def one_hot_embedding(labels, num_classes=16):
    device = labels.device
    y = torch.eye(num_classes, device=device)
    return y[labels]


class DynamicEdgeConvNet(torch.nn.Module):
    def __init__(
            self,
            out_channels,
            k,
            stn_k=20,
            aggr="max",
            transform_net=True,
            sample_size=2048,
            multiclass=False,
            s3dis=False
    ):
        super(DynamicEdgeConvNet, self).__init__()
        self.transform_net = transform_net
        self.sample_size = sample_size
        self.multiclass = multiclass
        self.s3dis = s3dis
        if self.transform_net:
            self.transform_net = Transform_Net(stn_k=stn_k)
        self.init_features = 3
        if self.s3dis:
            self.init_features = 9
        self.k = k
        self.conv1 = DynamicEdgeConv(
            MLP([2 * self.init_features, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(
            MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(
            MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        mlp_in_features = 1024
        if self.multiclass:
            mlp_in_features += 3 * 64 + 16
        if self.s3dis:
            mlp_in_features += 3 * 64

        self.mlp = Seq(
            MLP([mlp_in_features, 256]),
            Dropout(0.5),
            MLP([256, 128]),
            Dropout(0.5),
            nn.Linear(128, out_channels),
        )

    def forward(self, pos, batch, data):
        device = pos.device

        if self.transform_net:
            pos = self.transform_net.transform(pos, batch)
            x = pos
        if self.s3dis:
            x = torch.cat([pos, x], dim=1)

        if self.multiclass or self.s3dis:
            origbatch = batch.clone()

        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        activations = torch.cat([x1, x2, x3], dim=1)
        out = self.lin1(activations)

        if self.multiclass:
            cat = data.category.to(device)
            out = global_max_pool(out, origbatch)
            out = out.repeat_interleave(repeats=len(x1) // len(out), dim=0)
            onehot = one_hot_embedding(cat).to(device)
            onehot = onehot.repeat_interleave(repeats=len(x1) // len(onehot), dim=0)

            out = torch.cat([x1, x2, x3, out, onehot], dim=1)
        if self.s3dis:
            out = global_max_pool(out, origbatch)
            out = out.repeat_interleave(repeats=self.sample_size, dim=0)
            out = torch.cat([out, activations], dim=1)

        out = self.mlp(out)
        return F.log_softmax(out, dim=1)