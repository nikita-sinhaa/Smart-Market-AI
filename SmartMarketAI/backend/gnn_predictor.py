import torch
from backend.gnn_fraud_detector import GNNFraudDetector, build_graph

def predict_gnn(df):
    data = build_graph(df)
    model = GNNFraudDetector(in_channels=1, hidden_channels=16)
    model.load_state_dict(torch.load("models/gnn_fraud_model.pt"))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = torch.argmax(out, dim=1)
    return preds.numpy()
