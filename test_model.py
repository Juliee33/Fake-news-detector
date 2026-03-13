from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "models/distilbert_fake_news_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Breaking news: Scientists discover water on Mars."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

outputs = model(**inputs)
prediction = torch.argmax(outputs.logits)

print("Prediction:", prediction.item())