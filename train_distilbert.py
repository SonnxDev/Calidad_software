# train_distilbert.py
import os
# (Opcional) silenciar warning de symlinks en Windows para el cache del hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import glob
import sys
from dataclasses import dataclass
from typing import List, Dict
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import evaluate

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# ===================== CONFIG =====================
DATA_DIR = "data"   # carpeta que contiene tus CSV (uno por app)
MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "outputs/distilbert-mx-sent"

SEED = 42
TEST_SIZE = 0.15    # 15% test
VAL_SIZE = 0.15     # 15% valid del restante (≈ 12.75% global)
MAX_LENGTH = 192
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2  # early stopping (si la versión lo soporta)
# ===================================================

LABEL2ID: Dict[str, int] = {"negativo": 0, "neutro": 1, "positivo": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

def _print_env():
    import transformers, datasets
    print("=== RUNTIME INFO ===")
    print("PYTHON:", sys.version)
    print("EXECUTABLE:", sys.executable)
    print("TRANSFORMERS:", transformers.__version__)
    print("DATASETS:", datasets.__version__)
    print("PYTORCH:", torch.__version__)
    print("CUDA available?:", torch.cuda.is_available())
    print("====================")

def load_all_csv(data_dir: str) -> pd.DataFrame:
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron CSV en '{data_dir}'. "
                                f"Coloca ahí tus archivos generados por el scraper.")
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # aseguramos columnas clave
        for col in ["text", "sentimiento", "appId"]:
            if col not in df.columns:
                raise ValueError(f"El CSV '{p}' no contiene la columna requerida '{col}'.")
        df = df[df["text"].astype(str).str.strip().ne("")]
        df = df[df["sentimiento"].isin(LABEL2ID.keys())]
        frames.append(df[["text", "sentimiento", "appId"]])
    if not frames:
        raise ValueError("No hay filas válidas tras filtrar texto y sentimiento.")
    return pd.concat(frames, ignore_index=True)

def prepare_splits(df_all: pd.DataFrame):
    df_all = df_all.dropna(subset=["text", "sentimiento"]).copy()
    df_all["label"] = df_all["sentimiento"].map(LABEL2ID).astype(int)

    # Split estratificado
    train_df, test_df = train_test_split(
        df_all, test_size=TEST_SIZE, stratify=df_all["label"], random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, stratify=train_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def tokenize_datasets(train_df, val_df, test_df, tokenizer):
    ds_train = Dataset.from_pandas(train_df)
    ds_val   = Dataset.from_pandas(val_df)
    ds_test  = Dataset.from_pandas(test_df)
    dataset = DatasetDict(train=ds_train, validation=ds_val, test=ds_test)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    encoded = dataset.map(preprocess, batched=True, desc="Tokenizing")
    return encoded

def compute_metrics_builder():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "precision_macro": prec.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": rec.compute(predictions=preds, references=labels, average="macro")["recall"],
        }
    return compute_metrics

class WeightedTrainer(Trainer):
    """Trainer con CrossEntropy ponderada por clase."""
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Sobreescribe compute_loss para usar ponderación por clase.
        Ahora acepta 'num_items_in_batch' a través de kwargs.
        """
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def build_training_args():
    """
    Construye TrainingArguments de forma adaptativa y consistente:
    - Si existen evaluation_strategy y save_strategy => ambos "epoch".
    - Define metric_for_best_model (f1_macro si está disponible; si no, eval_loss).
    - Activa load_best_model_at_end SOLO si las estrategias coinciden.
    """
    kwargs = dict(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        seed=SEED,
        report_to=[],   # evita W&B por defecto
    )

    sig = signature(TrainingArguments.__init__).parameters

    # Preferidos (v4.x)
    has_eval_strategy = "evaluation_strategy" in sig
    has_save_strategy = "save_strategy" in sig
    if has_eval_strategy:
        kwargs["evaluation_strategy"] = "epoch"
    if has_save_strategy:
        kwargs["save_strategy"] = "epoch"

    # Métrica para best model / early stopping
    if "metric_for_best_model" in sig:
        # Usaremos f1_macro (que definimos en compute_metrics). Si no existiera, eval_loss.
        kwargs["metric_for_best_model"] = "f1_macro"

    if "greater_is_better" in sig:
        kwargs["greater_is_better"] = True  # para f1_macro
    if "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = WARMUP_RATIO
    if "gradient_accumulation_steps" in sig:
        kwargs["gradient_accumulation_steps"] = GRAD_ACCUM
    if "fp16" in sig and torch.cuda.is_available():
        kwargs["fp16"] = True

    # Legacy (v2/v3) fallback
    if not has_eval_strategy or not has_save_strategy:
        # Estrategia por steps, consistentes
        if "evaluate_during_training" in sig:
            kwargs["evaluate_during_training"] = True
        step_val = 500
        if "eval_steps" in sig:
            kwargs["eval_steps"] = step_val
        if "save_steps" in sig:
            kwargs["save_steps"] = step_val
        if "warmup_steps" in sig and "warmup_ratio" not in kwargs:
            kwargs["warmup_steps"] = 500
        # No activamos load_best_model_at_end en legacy
    else:
        # En v4, sólo activamos load_best_model_at_end si eval y save NO son "no"
        kwargs["load_best_model_at_end"] = True

    ta = TrainingArguments(**kwargs)

    # Salvaguarda final: si por alguna razón eval/save no están activos, ajusta métrica y best model
    eval_str = str(getattr(ta, "evaluation_strategy", "no")).lower()
    save_str = str(getattr(ta, "save_strategy", "no")).lower()
    if eval_str == "no" or save_str == "no" or eval_str != save_str:
        # Desactivar best model y usar eval_loss si fuera necesario
        if hasattr(ta, "load_best_model_at_end"):
            ta.load_best_model_at_end = False
        if hasattr(ta, "metric_for_best_model") and (ta.metric_for_best_model is None):
            ta.metric_for_best_model = "eval_loss"
        if hasattr(ta, "greater_is_better") and ta.metric_for_best_model == "eval_loss":
            ta.greater_is_better = False

    return ta

# Usa el builder adaptativo
training_args = build_training_args()

# Early Stopping si está disponible en tu versión
callbacks = []
try:
    from transformers import EarlyStoppingCallback
    # Condición: evaluación activa y métrica definida
    eval_active = str(getattr(training_args, "evaluation_strategy", "no")).lower() != "no"
    metric_defined = getattr(training_args, "metric_for_best_model", None) is not None
    if eval_active and metric_defined:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    else:
        print("Aviso: EarlyStoppingCallback omitido (no hay evaluación activa o falta metric_for_best_model).")
except Exception:
    print("Aviso: EarlyStoppingCallback no disponible en esta versión; se omite.")

def main():
    _print_env()
    set_seed(SEED)

    # 1) Cargar datos
    df_all = load_all_csv(DATA_DIR)
    print(f"Total filas (tras filtro): {len(df_all)}")

    # 2) Splits
    train_df, val_df, test_df = prepare_splits(df_all)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 3) Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    # 4) Tokenización
    encoded = tokenize_datasets(train_df, val_df, test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Pesos por clase
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 6) Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7) TrainingArguments (adaptativo a versión)
    training_args = build_training_args()

    # 8) Métricas
    compute_metrics = compute_metrics_builder()

    # 9) Trainer con loss ponderada
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 10) Entrenar
    trainer.train()

    # 11) Evaluar en test
    metrics_test = trainer.evaluate(encoded["test"])
    print("Test metrics:", metrics_test)

    # 12) Guardar modelo y tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 13) Demo de inferencia
    LABELS = ["negativo", "neutro", "positivo"]

    def predict(texts: List[str]):
        tokens = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        model.eval()
        with torch.no_grad():
            logits = model(**tokens).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        return [LABELS[p] for p in preds]

    ejemplos = [
        "La app es excelente, muy rápida",
        "Se cierra a cada rato, pésima experiencia",
        "Está bien, cumple lo justo"
    ]
    print("Demo:", list(zip(ejemplos, predict(ejemplos))))

if __name__ == "__main__":
    main()
