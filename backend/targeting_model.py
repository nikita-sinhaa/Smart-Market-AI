from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
from utils.data_preprocessing import load_data, preprocess
import os

def train_targeting_model(data_path="data/sample_campaign.csv"):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(df, target_col="converted")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    dump(model, "models/targeting_model.pkl")
    print(f"✅ Targeting model trained. Accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_targeting_model()
