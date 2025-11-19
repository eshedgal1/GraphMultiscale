import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv


class GCNII(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(50, hidden_channels))
        self.lins.append(Linear(hidden_channels, 121))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        """Applies Xavier Uniform Initialization to all layers."""
        for conv in self.convs:
            if hasattr(conv, 'lin'):
                torch.nn.init.xavier_uniform_(conv.lin.weight)

        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight)

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x