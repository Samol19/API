from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm
import spacy
import torch
import torch.nn.functional as F
import joblib
import numpy as np

#Para manejar cors
from fastapi.middleware.cors import CORSMiddleware

nlp = spacy.load('en_core_web_sm')

app = FastAPI()

#Cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las solicitudes de origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)





# Directorios de archivos
model_path = "Results/Model"
tokenizer_path = "Results/Tokenizer"

# Cargar modelo y tokenizer de Distil Bert
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

# Cargar modelos y vectorizadores de Scikit
mnb = joblib.load('Results-Scikit/Models/Multinomial NB.pkl')
rf = joblib.load('Results-Scikit/Models/Random Forest d_50.pkl')
rl = joblib.load('Results-Scikit/Models/Regresion Logistica.pkl')
#svm = joblib.load('Results-Scikit/Models/SVM lineal.pkl')
vectorizer = joblib.load('Results-Scikit/Vectorizer/vectorizer.pkl')

# Creamos un diccionario con los modelos
models = {
    "MultinomialNB": mnb,
    "RandomForestd_50": rf,
    "RegresionLogistica": rl
    #"SVM lineal": svm
}

# Definir el esquema del input usando Pydantic
class TextRequest(BaseModel):
    text: str

# Creamos la funcion para normalizar los textos
def normalize(sentenses):
    """normalizamos la lista de frases y devolvemos la misma lista de frases normalizada"""
    for index, sentense in enumerate(tqdm(sentenses)):
        sentense = nlp(sentense.lower()) # Paso la frase a minúsculas y a un Doc de Spacy
        sentenses[index] = " ".join([word.lemma_ for word in sentense if (not word.is_punct)
                                     and (len(word.text) > 2) and (not word.is_stop)])
    return sentenses


# Definimos la ruta de la API para DistilBert
@app.post("/predict")
async def predict(request: TextRequest):
    # Tokenizar el input
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1).item()
        confidence = torch.max(probabilities, dim=-1).values.item()

    if predictions == 0:
        predictions = "Negative"
    elif predictions == 1:
        predictions = "Neutral"
    elif predictions == 2:
        predictions = "Positive"


    return {"prediction": predictions, "confidence": confidence}

# Definimos la ruta de la API para los modelos de Scikit
@app.post("/predict_scikit")
async def predict_scikit(request: TextRequest):
    # Nomalizamos el texto
    texts = normalize([request.text])

    # Vectorizamos el texto
    x = vectorizer.transform(texts)

    # Predecimos la clase por modelo
    predictions = {}
    for key, model in models.items():
        prob = model.predict_proba(x)[0]
        pred = model.predict(x)[0]
        confidence = np.max(prob)
        predictions[key] = {"prediction": pred, "confidence": confidence}

    return predictions


# Comando para correr la api: uvicorn Model_API:app --reload