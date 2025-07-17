import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

class GNNFraudDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNFraudDetector, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def build_graph(df):
    edge_index = []
    node_features = []
    node_labels = []
    node_map = {}
    node_count = 0

    for i, row in df.iterrows():
        user = f"user_{row['age']}_{row['country']}"
        device = f"device_{row['device_type']}"
        ad = f"ad_{i}"

        fraud_label = row.get("converted", 0)

        for node in [user, device, ad]:
            if node not in node_map:
                node_map[node] = node_count
                node_features.append([1.0])
                node_labels.append(fraud_label if 'ad' in node else 0)
                node_count += 1

        edge_index.append([node_map[user], node_map[device]])
        edge_index.append([node_map[device], node_map[ad]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def train_gnn(data):
    model = GNNFraudDetector(in_channels=1, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    data.train_mask = torch.rand(data.num_nodes) < 0.8
    data.test_mask = ~data.train_mask

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = evaluate_gnn(model, data)
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Test Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "models/gnn_fraud_model.pt")
    print("✅ GNN model trained and saved as 'models/gnn_fraud_model.pt'")
    return model

def evaluate_gnn(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc
