from sklearn.linear_model import LinearRegression
from joblib import dump
from utils.data_preprocessing import load_data, preprocess
import os

def train_pricing_model(data_path="data/sample_campaign.csv"):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(df, target_col="bid_price")
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    dump(model, "models/pricing_model.pkl")
    print(f"✅ Pricing model trained. R² score: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_pricing_model()
