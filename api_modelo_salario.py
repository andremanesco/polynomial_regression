from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd

# Criar uma instância do FastAPI
app = FastAPI()

# Criar classe com os dados de entrada que virão no request body com os tipos esperados
class request_body(BaseModel):
    tempo_na_empresa: int
    nivel_na_empresa: int

# Carregar modelo para realizar a predição
model_poly = joblib.load('./modelo_salario.pkl')

@app.post('/predict')
def predict(data: request_body):
    input_features = {
    "tempo_na_empresa": data.tempo_na_empresa,
    "nivel_na_empresa": data.nivel_na_empresa
}

    pred_df = pd.DataFrame(input_features, index=[1])

    # Predição
    y_pred = model_poly.predict(pred_df)[0].astype(float)

    return {'salario_em_reais': y_pred.tolist()}