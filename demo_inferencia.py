import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el modelo y el tokenizador entrenado (asegúrate de que estén guardados)
MODEL_PATH = "outputs/distilbert-mx-sent"  # Ruta del modelo entrenado

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Función para predecir el sentimiento de una reseña
def predict_sentiment(text):
    # Tokenización
    inputs = tokenizer(text, truncation=True, padding=True, max_length=192, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inferencia
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    # Mapeo de etiquetas
    LABELS = ["negativo", "neutro", "positivo"]
    return LABELS[preds[0]]


# Función que se ejecuta cuando el botón es presionado
def on_predict_button_click():
    review_text = text_entry.get("1.0", "end-1c")  # Obtener texto completo de Text widget

    # Verificar que no haya números en la reseña
    if any(char.isdigit() for char in review_text):
        messagebox.showerror("Error", "La reseña no debe contener números.")
        return

    if not review_text.strip():  # Si está vacío o solo espacios
        messagebox.showerror("Error", "Por favor, ingrese una reseña.")
        return

    # Predecir el sentimiento de la reseña
    sentiment = predict_sentiment(review_text)

    # Actualizar el color del texto de acuerdo con el sentimiento
    if sentiment == "positivo":
        result_label.config(text=f"Sentimiento de la reseña: {sentiment}", fg="green")
    elif sentiment == "neutro":
        result_label.config(text=f"Sentimiento de la reseña: {sentiment}", fg="orange")
    else:
        result_label.config(text=f"Sentimiento de la reseña: {sentiment}", fg="red")


# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Clasificador de Sentimiento de Reseñas")

# Etiqueta de instrucciones
instruction_label = tk.Label(root, text="Ingrese una reseña para clasificar su sentimiento (sin números):",
                             font=("Helvetica", 12))
instruction_label.pack(pady=10)

# Campo de texto para escribir la reseña (más grande)
text_entry = tk.Text(root, width=50, height=10, font=("Helvetica", 12))  # más grande que Entry
text_entry.pack(pady=10)

# Botón para ejecutar la demo con color de fondo
predict_button = tk.Button(root, text="Clasificar Sentimiento", command=on_predict_button_click, bg="#4CAF50",
                           fg="white", font=("Helvetica", 12))
predict_button.pack(pady=10)

# Etiqueta para mostrar el resultado de la predicción
result_label = tk.Label(root, text="Sentimiento de la reseña: ", font=("Helvetica", 12))
result_label.pack(pady=20)

# Iniciar la aplicación
root.mainloop()