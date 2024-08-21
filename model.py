import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class gcn(torch.nn.Module):
    def __init__(self, num_features,num_classes,hidden):
        super(gcn, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, dropout):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
