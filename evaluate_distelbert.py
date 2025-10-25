import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ===================== CONFIG =====================
CSV_PATH = "review/reviews_test.csv"  # Ruta a tu CSV de evaluación
MODEL_PATH = "outputs/distilbert-mx-sent"  # Modelo entrenado
MAX_LENGTH = 192
BATCH_SIZE = 16  # Procesar en lotes para eficiencia
# ===================================================

LABELS = ["negativo", "neutro", "positivo"]
LABEL2ID = {"negativo": 0, "neutro": 1, "positivo": 2}


def load_data(csv_path):
    """
    Carga el CSV. Columnas esperadas: text, sentimiento
    Columnas opcionales: score, appId, userName, etc.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Columnas detectadas: {', '.join(df.columns)}")

    # Verificar columnas requeridas
    required_cols = ["text", "sentimiento"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"El CSV debe contener la columna '{col}'")

    # Detectar columna de estrellas
    if "score" in df.columns:
        print(f"✓ Columna de estrellas detectada: 'score'")
        print(f"  Distribución de scores:")
        print(df["score"].value_counts().sort_index())

    # Detectar otras columnas útiles
    if "appId" in df.columns:
        print(f"✓ Detectadas {df['appId'].nunique()} apps diferentes")

    # Filtrar filas válidas
    original_len = len(df)
    df = df.dropna(subset=["text", "sentimiento"])
    df = df[df["text"].astype(str).str.strip().ne("")]
    df = df[df["sentimiento"].isin(LABEL2ID.keys())]

    if len(df) < original_len:
        print(f"⚠️  Se filtraron {original_len - len(df)} filas inválidas")

    print(f"\nTotal de reseñas válidas: {len(df)}")
    print(f"Distribución de sentimientos:")
    print(df["sentimiento"].value_counts().sort_index())

    return df


def predict_batch(texts, tokenizer, model, device):
    """
    Predice sentimientos en lotes para eficiencia
    """
    tokens = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**tokens).logits

    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [LABELS[p] for p in preds]


def evaluate_model(df, tokenizer, model, device):
    """
    Evalúa el modelo en todo el dataset
    """
    texts = df["text"].tolist()
    true_labels = df["sentimiento"].tolist()

    # Predicciones en lotes
    predictions = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nRealizando predicciones en {total_batches} lotes...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_preds = predict_batch(batch, tokenizer, model, device)
        predictions.extend(batch_preds)

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"Procesados {i + len(batch)}/{len(texts)} textos...")

    return true_labels, predictions


def calculate_metrics(true_labels, predictions):
    """
    Calcula métricas de evaluación
    """
    # Convertir labels a IDs numéricos para algunas métricas
    true_ids = [LABEL2ID[label] for label in true_labels]
    pred_ids = [LABEL2ID[label] for label in predictions]

    # Métricas generales
    accuracy = accuracy_score(true_ids, pred_ids)

    # Métricas macro (promedio de todas las clases)
    precision_macro = precision_score(true_ids, pred_ids, average='macro', zero_division=0)
    recall_macro = recall_score(true_ids, pred_ids, average='macro', zero_division=0)
    f1_macro = f1_score(true_ids, pred_ids, average='macro', zero_division=0)

    # Métricas weighted (ponderadas por soporte de clase)
    precision_weighted = precision_score(true_ids, pred_ids, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_ids, pred_ids, average='weighted', zero_division=0)
    f1_weighted = f1_score(true_ids, pred_ids, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN")
    print("=" * 60)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"\nMétricas Macro (promedio simple):")
    print(f"  Precision:        {precision_macro:.4f}")
    print(f"  Recall:           {recall_macro:.4f}")
    print(f"  F1-Score:         {f1_macro:.4f}")
    print(f"\nMétricas Weighted (ponderadas por clase):")
    print(f"  Precision:        {precision_weighted:.4f}")
    print(f"  Recall:           {recall_weighted:.4f}")
    print(f"  F1-Score:         {f1_weighted:.4f}")

    # Reporte detallado por clase
    print(f"\n{'-' * 60}")
    print("REPORTE POR CLASE")
    print("-" * 60)
    print(classification_report(
        true_ids,
        pred_ids,
        target_names=LABELS,
        digits=4,
        zero_division=0
    ))

    # Matriz de confusión
    cm = confusion_matrix(true_ids, pred_ids)
    print("-" * 60)
    print("MATRIZ DE CONFUSIÓN")
    print("-" * 60)
    print("Predicho →    ", "  ".join(f"{l:>10}" for l in LABELS))
    print("Real ↓")
    for i, label in enumerate(LABELS):
        print(f"{label:>10}    ", "  ".join(f"{cm[i][j]:>10}" for j in range(len(LABELS))))

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }


def show_error_examples(df, true_labels, predictions, n_examples=5):
    """
    Muestra ejemplos de predicciones incorrectas
    """
    print(f"\n{'=' * 60}")
    print(f"EJEMPLOS DE ERRORES (primeros {n_examples})")
    print("=" * 60)

    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            row = df.iloc[i]
            error_info = {
                'idx': i,
                'true': true,
                'pred': pred,
                'text': row["text"],
                'score': row.get("score", None),
                'appId': row.get("appId", None)
            }
            errors.append(error_info)

    print(f"Total de errores: {len(errors)} de {len(true_labels)} ({len(errors) / len(true_labels) * 100:.2f}%)")
    print()

    for i, error in enumerate(errors[:n_examples], 1):
        print(f"Error {i}:")
        text = error['text']
        print(f"  Texto:      {text[:100]}..." if len(text) > 100 else f"  Texto:      {text}")
        if error['score'] is not None:
            print(f"  Score:      {error['score']} estrellas")
        if error['appId'] is not None:
            print(f"  App:        {error['appId']}")
        print(f"  Real:       {error['true']}")
        print(f"  Predicho:   {error['pred']}")
        print()

    # Análisis adicional si hay score
    if 'score' in df.columns:
        print(f"{'-' * 60}")
        print("ANÁLISIS DE ERRORES POR SCORE")
        print("-" * 60)
        df_errors = df.iloc[[e['idx'] for e in errors]].copy()
        df_errors['prediccion'] = [e['pred'] for e in errors]
        print(df_errors.groupby('score').size().sort_index().to_string())
        print()


def main():
    print("=" * 60)
    print("EVALUACIÓN DE MODELO DISTILBERT")
    print("=" * 60)

    # 1. Cargar datos
    print(f"\n1. Cargando datos desde: {CSV_PATH}")
    df = load_data(CSV_PATH)

    # 2. Cargar modelo
    print(f"\n2. Cargando modelo desde: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Dispositivo: {device}")

    # 3. Evaluar modelo
    print(f"\n3. Evaluando modelo...")
    true_labels, predictions = evaluate_model(df, tokenizer, model, device)

    # 4. Calcular métricas
    print(f"\n4. Calculando métricas...")
    metrics = calculate_metrics(true_labels, predictions)

    # 5. Mostrar ejemplos de errores
    show_error_examples(df, true_labels, predictions, n_examples=5)

    # 6. Guardar resultados (opcional)
    results_df = df.copy()
    results_df["prediccion"] = predictions
    results_df["correcto"] = [t == p for t, p in zip(true_labels, predictions)]
    output_path = "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Resultados guardados en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()