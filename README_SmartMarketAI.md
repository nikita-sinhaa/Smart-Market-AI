# 🧠 SmartMarketAI

**An Intelligent Ad Bidding, Pricing Optimization, and Fraud Detection System**  
Built with Python, Machine Learning, and Graph Neural Networks  
> 📍 Designed to mirror real-world algorithms engineering roles in ad-tech & marketing intelligence

---

## 🚀 Project Summary

SmartMarketAI is a complete machine learning pipeline and interactive dashboard that simulates a marketing platform's core decision-making intelligence.

It features:
- 🎯 Conversion Prediction using Gradient Boosting
- 💸 Dynamic Bid Price Optimization via Regression
- 🕵️‍♀️ Fraud Detection via both Isolation Forest and GNN (Graph Neural Network)
- 📊 Streamlit-based UI for campaign simulation and experimentation

---

## 🧩 Features Breakdown

| Module                  | Technique                       | Purpose                                      |
|------------------------|----------------------------------|----------------------------------------------|
| Targeting              | GradientBoostingClassifier       | Predict which users are most likely to convert |
| Pricing Optimization   | Linear Regression                | Recommend optimal bid price for impressions  |
| Classic Fraud Detection| Isolation Forest                 | Detect outlier click behaviors               |
| Advanced Fraud GNN     | PyTorch Geometric + GCN          | Learn fraud patterns from graph structures   |

---

## 🖥️ Tech Stack

- **Python 3.10**
- `scikit-learn`, `joblib` for ML models
- `Streamlit` for UI
- `PyTorch Geometric` for GNN-based fraud detection
- `pandas`, `numpy` for data wrangling

---

## 🧪 Sample Data

The dataset simulates 1,000 ad campaign logs with the following features:

- `age`, `country`, `device_type`, `ad_click_rate`
- `converted`: 1 if converted, 0 otherwise
- `bid_price`: actual bid used

You can replace this with real ad logs like Criteo or Avazu datasets for real-world tuning.

---

## 📂 Project Structure

```
SmartMarketAI/
│
├── backend/
│   ├── targeting_model.py        # Classification (conversion)
│   ├── pricing_optimizer.py      # Regression (bid prices)
│   ├── fraud_detector.py         # Isolation Forest fraud detector
│   ├── gnn_fraud_detector.py     # GCN-based graph learning
│   └── gnn_predictor.py          # GNN inference
│
├── data/
│   └── sample_campaign.csv       # Simulated dataset
│
├── frontend/
│   └── dashboard.py              # Streamlit UI
│
├── models/                       # .pkl and .pt model files
│
├── utils/
│   └── data_preprocessing.py
│
├── app.py                        # Hugging Face entrypoint
├── render.yaml                   # Render deployment config
├── SpaceConfig                   # Hugging Face Space config
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. 📤 Upload your campaign dataset in the UI.
2. 🧠 Select and run models for targeting, pricing, or fraud.
3. 🕵️‍♀️ GNN fraud predictions highlight suspicious entities via relational reasoning.
4. 📈 Results shown live with KPIs and summary statistics.

---

## 🧪 Running Locally

```bash
git clone https://github.com/yourusername/SmartMarketAI.git
cd SmartMarketAI
pip install -r requirements.txt

# Optional: train models
python backend/targeting_model.py
python backend/pricing_optimizer.py
python backend/fraud_detector.py

# Launch dashboard
streamlit run frontend/dashboard.py
```

---

## 🧠 GNN Supervised Fraud Detection (Optional)

```python
from backend.gnn_fraud_detector import build_graph, train_gnn
import pandas as pd

df = pd.read_csv("data/sample_campaign.csv")
graph_data = build_graph(df)
train_gnn(graph_data)
```

---

## 🌐 Deployment Options

| Platform      | Status     | Description                                  |
|---------------|------------|----------------------------------------------|
| Hugging Face  | ✅ Ready    | Drag and drop all files; uses `app.py`       |
| Render        | ✅ Ready    | Connect repo and auto-deploy with `render.yaml` |

---

## 👩‍💻 About the Author

**Nikita Sinha**  
🔬 Embedded Software & Algorithms Engineer  
🎓 MS in Electrical & Computer Engineering, Purdue University  
🔗 [LinkedIn](https://www.linkedin.com/in/nikita-sinhaa/) • [GitHub](https://github.com/nikita-sinhaa)

---

## 📌 Why This Project?

> Created as part of my portfolio to demonstrate my ability to solve high-impact problems in **fraud detection**, **pricing optimization**, and **real-time targeting** using applied machine learning and scalable architecture.

This project is designed to align with roles like:
- Senior Algorithms Engineer
- Applied Scientist (Marketing Optimization)
- Machine Learning Engineer in Ad-Tech

---

## 🧠 Future Work

- 🧬 Integrate Real Datasets (e.g., Criteo logs)
- 📈 Add model monitoring + drift detection
- 📉 Reinforcement learning for dynamic bidding
- 🕸️ Node2Vec or GAT for better GNN predictions

---

## 📬 Contact

Want to collaborate or hire me?  
📧 Reach me at: [email@example.com]  
💬 Or drop a message on [LinkedIn](https://www.linkedin.com/in/nikita-sinhaa/)

---

> _"Where data meets decision, and structure meets scale—this is where I thrive."_ – Nikita