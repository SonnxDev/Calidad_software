from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carga tu modelo entrenado (carpeta generada por el script de entrenamiento)
MODEL_PATH = "outputs/distilbert-mx-sent"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = ["negativo", "neutro", "positivo"]

def predict(texts):
    tokens = tokenizer(texts, truncation=True, padding=True, max_length=192, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        logits = model(**tokens).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [LABELS[p] for p in preds]

# --- DEMO ---
ejemplos = [
    "Fémur",
    "Está desactualizada",
    "Es una aplicación buena, pero tiene algunos fallos"
]
for t, pred in zip(ejemplos, predict(ejemplos)):
    print(f"{pred.upper():<10} → {t}")
