from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn 
import joblib

# Criar uma instancia do FastAPI

app = FastAPI()

# Criar uma classe para o request body
class request_body(BaseModel):
    horas_estudo: float
    
# Carregar o modelo

modelo_pontuacao = joblib.load('modelo_regressao_linear.pkl')

@app.post('/predict')
def predict(data: request_body):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]
    # Fazer a predição
    Y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)
    return {'pontuacao_teste' : Y_pred.tolist()}
