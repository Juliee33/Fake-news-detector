import os
import time
import pandas as pd

from .predict import predict_news
from .predict_distilbert import predict_fake_news

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")

df = pd.read_csv(data_path)

# ---------------------------------------------------------
# Accuracy / Confusion Matrix Evaluation (500 samples)
# ---------------------------------------------------------

results = []
sample = df.sample(500, random_state=42)

for _, row in sample.iterrows():

    text = row["text"]
    true_label = row["label"]

    true_label_text = "Real News" if true_label == 1 else "Fake News"

    svm_pred, _ = predict_news(text)
    bert_pred, _ = predict_fake_news(text)

    results.append({
        "True Label": true_label_text,
        "SVM Prediction": svm_pred,
        "DistilBERT Prediction": bert_pred
    })

results_df = pd.DataFrame(results)

print(results_df.head())

# ---------------------------------------------------------
# Save prediction comparison table
# ---------------------------------------------------------

results_path = os.path.join(BASE_DIR, "results", "model_comparison_table.csv")
results_df.to_csv(results_path, index=False)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

y_true = results_df["True Label"]
svm_pred = results_df["SVM Prediction"]
bert_pred = results_df["DistilBERT Prediction"]

print("\nSVM Metrics")
print("Accuracy:", accuracy_score(y_true, svm_pred))
print("Precision:", precision_score(y_true, svm_pred, pos_label="Fake News"))
print("Recall:", recall_score(y_true, svm_pred, pos_label="Fake News"))
print("F1:", f1_score(y_true, svm_pred, pos_label="Fake News"))

print("\nDistilBERT Metrics")
print("Accuracy:", accuracy_score(y_true, bert_pred))
print("Precision:", precision_score(y_true, bert_pred, pos_label="Fake News"))
print("Recall:", recall_score(y_true, bert_pred, pos_label="Fake News"))
print("F1:", f1_score(y_true, bert_pred, pos_label="Fake News"))

# ---------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------

svm_cm = confusion_matrix(y_true, svm_pred)
bert_cm = confusion_matrix(y_true, bert_pred)

print("\nSVM Confusion Matrix")
print(svm_cm)

print("\nDistilBERT Confusion Matrix")
print(bert_cm)

comparison_df = pd.DataFrame({
    "Actual Class": ["Fake News", "Real News"],
    "SVM Pred Fake": [svm_cm[0][0], svm_cm[1][0]],
    "SVM Pred Real": [svm_cm[0][1], svm_cm[1][1]],
    "DistilBERT Pred Fake": [bert_cm[0][0], bert_cm[1][0]],
    "DistilBERT Pred Real": [bert_cm[0][1], bert_cm[1][1]]
})

comparison_path = os.path.join(BASE_DIR, "results", "confusion_matrix_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)




# ---------------------------------------------------------
# Latency Experiment (50 samples)
# ---------------------------------------------------------

latency_sample = df.sample(50, random_state=1)

svm_times = []
bert_times = []
latency_records = []

for i, (_, row) in enumerate(latency_sample.iterrows()):

    text = row["text"]

    # SVM latency
    start = time.time()
    predict_news(text)
    svm_time = time.time() - start

    # DistilBERT latency
    start = time.time()
    predict_fake_news(text)
    bert_time = time.time() - start

    svm_times.append(svm_time)
    bert_times.append(bert_time)

    latency_records.append({
        "Run": i + 1,
        "SVM Latency (seconds)": svm_time,
        "DistilBERT Latency (seconds)": bert_time
    })

# ---------------------------------------------------------
# Save latency runs (for graphs)
# ---------------------------------------------------------

latency_runs_df = pd.DataFrame(latency_records)

latency_runs_path = os.path.join(BASE_DIR, "results", "latency_runs.csv")
latency_runs_df.to_csv(latency_runs_path, index=False)

# ---------------------------------------------------------
# Save average latency
# ---------------------------------------------------------

avg_svm_latency = sum(svm_times) / len(svm_times)
avg_bert_latency = sum(bert_times) / len(bert_times)

latency_df = pd.DataFrame({
    "Model": ["Linear SVM", "DistilBERT"],
    "Average Latency (seconds)": [avg_svm_latency, avg_bert_latency]
})

latency_path = os.path.join(BASE_DIR, "results", "latency_comparison.csv")
latency_df.to_csv(latency_path, index=False)

print("\nLatency Results")
print(latency_df)

print("\nLatency runs saved to latency_runs.csv")
print("\nAll evaluation tables saved in results folder.")