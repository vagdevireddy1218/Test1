from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model = mlflow.pyfunc.load_model(
    "models:/iris_model/Production"
)

@app.get("/")
def home():
    return {"message": "MLflow FastAPI Service"}

@app.post("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):

    df = pd.DataFrame([{
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width
    }])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }