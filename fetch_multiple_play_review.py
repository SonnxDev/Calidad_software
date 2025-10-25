import time
import re
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from google_play_scraper import reviews, Sort

# ===================== CONFIG =====================
APP_IDS = [
    "com.whatsapp",
    "com.spotify.music",
    "com.netflix.mediaclient",
    "com.instagram.android",
    "com.twitter.android"
]
TARGET_POSITIVE = 350  # 350 positivas
TARGET_NEGATIVE = 350  # 350 negativas
TARGET_NEUTRAL = 300   # 300 neutras
LANG = "es"
COUNTRY = "pe"
SORTING = Sort.NEWEST
BATCH_SIZE = 200
PAUSE_SECONDS = 1.2
OUT_PREFIX = f"ds_calidad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ==================================================

def basic_clean(text: Optional[str]) -> str:
    """Limpieza mínima: sin eliminar puntuación/stopwords (solo espacios/HTML)."""
    if not text:
        return ""
    # eliminar tags muy simples de HTML si alguna versión del scraper devolviera
    text = re.sub(r"<[^>]+>", " ", text)
    # normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text

def map_score_to_sentiment(score: Optional[int]) -> Optional[str]:
    """Pre-anotado según el plan (1–2 neg, 3 neu, 4–5 pos)."""
    if score is None:
        return None
    try:
        s = int(score)
    except Exception:
        return None
    if s <= 2:
        return "negativo"
    if s == 3:
        return "neutro"
    return "positivo"  # 4–5

def fetch_reviews_one_app(app_id: str,
                          target_pos: int,
                          target_neg: int,
                          target_neutral: int,
                          lang: str,
                          country: str,
                          sort=Sort.NEWEST,
                          batch_size: int = 200,
                          pause_s: float = 1.2) -> List[Dict]:
    all_rows = []
    fetched_pos = fetched_neg = fetched_neutral = 0
    next_token: Optional[str] = None

    # Looper para seguir extrayendo hasta cumplir con los 350 positivas, 350 negativas y 300 neutras
    while fetched_pos < target_pos or fetched_neg < target_neg or fetched_neutral < target_neutral:
        to_fetch = min(batch_size, target_pos + target_neg + target_neutral - len(all_rows))
        try:
            result, next_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                count=to_fetch,
                continuation_token=next_token
            )
        except TypeError:
            result = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                count=to_fetch
            )
            next_token = None

        if not result:
            break

        # Filtramos las reseñas por sentimiento
        for review in result:
            sentiment = map_score_to_sentiment(review.get("score"))
            if sentiment == "positivo" and fetched_pos < target_pos:
                all_rows.append(review)
                fetched_pos += 1
            elif sentiment == "negativo" and fetched_neg < target_neg:
                all_rows.append(review)
                fetched_neg += 1
            elif sentiment == "neutro" and fetched_neutral < target_neutral:
                all_rows.append(review)
                fetched_neutral += 1

        # Verificamos si ya hemos alcanzado los objetivos
        if fetched_pos >= target_pos and fetched_neg >= target_neg and fetched_neutral >= target_neutral:
            break

        time.sleep(pause_s)

    return all_rows

def normalize_to_spec(rows: List[Dict], app_id: str) -> pd.DataFrame:
    """
    Transforma la salida del scraper al esquema EXACTO:
    id, userName, date, score, text, sentimiento, appId, id_anotador, notas, thumbsUp
    """
    # Campos que suelen venir del scraper:
    # reviewId, userName, at (datetime), score (1-5), content (texto), thumbsUpCount, ...
    df = pd.json_normalize(rows)

    # Renombrar / derivar según especificación
    df["id"] = df.get("reviewId")
    df["userName"] = df.get("userName")
    # Convertir fecha a YYYY-MM-DD (si se desea), manteniendo TZ si existiera
    if "at" in df.columns:
        df["date"] = pd.to_datetime(df["at"]).dt.date.astype(str)
    else:
        df["date"] = None

    df["score"] = df.get("score")
    # Limpieza mínima del texto
    src_text = df.get("content") if "content" in df.columns else None
    df["text"] = src_text.apply(basic_clean) if src_text is not None else ""

    # Pre-anotado (puede ser corregido luego por anotadores)
    df["sentimiento"] = df["score"].apply(map_score_to_sentiment)

    # appId fijo
    df["appId"] = app_id

    # Columnas de anotación humana (vacías por defecto; las llenarán luego)
    df["id_anotador"] = pd.NA
    df["notas"] = pd.NA

    # Votos de utilidad
    df["thumbsUp"] = df.get("thumbsUpCount", 0).fillna(0).astype(int)

    # Orden EXACTO de columnas
    cols = [
        "id", "userName", "date", "score", "text",
        "sentimiento", "appId", "id_anotador", "notas", "thumbsUp"
    ]
    # Asegurar que existan todas
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[cols].drop_duplicates(subset=["id", "userName", "date", "text"], keep="first")
    # Filtrar reseñas vacías (si las hubiera)
    df = df[df["text"].astype(str).str.strip().ne("")]
    return df

def main():
    for i, app in enumerate(APP_IDS, 1):
        print(f"[{i}/{len(APP_IDS)}] Descargando reseñas positivas, negativas y neutras de {app}")
        rows = fetch_reviews_one_app(
            app_id=app,
            target_pos=TARGET_POSITIVE,
            target_neg=TARGET_NEGATIVE,
            target_neutral=TARGET_NEUTRAL,
            lang=LANG,
            country=COUNTRY,
            sort=SORTING,
            batch_size=BATCH_SIZE,
            pause_s=PAUSE_SECONDS
        )
        df = normalize_to_spec(rows, app_id=app)

        # Guardar SOLO por app (como pide el plan de entregables)
        out_csv = f"{OUT_PREFIX}_{app.replace('.', '_')}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f" -> Guardado: {out_csv} ({len(df)} filas)")

        time.sleep(2)  # pausa entre apps para ser amable con la fuente

if __name__ == "__main__":
    main()