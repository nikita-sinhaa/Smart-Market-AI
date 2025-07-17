from sklearn.ensemble import IsolationForest
from joblib import dump
from utils.data_preprocessing import load_data
import os

def train_fraud_model(data_path="data/sample_campaign.csv"):
    # Load and preprocess data
    df = load_data(data_path)

    # Drop target columns if present to avoid label leakage
    features = df.drop(columns=[col for col in ['converted', 'bid_price'] if col in df.columns])

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(features)

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(model, "models/fraud_model.pkl")
    print("✅ Fraud detection model trained and saved as 'models/fraud_model.pkl'.")

if __name__ == "__main__":
    train_fraud_model()
