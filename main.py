#  Library imports
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pycaret.classification import load_model, predict_model

#  Create the app object
app = FastAPI()

# Load trained Pipeline
model = load_model('HD_model')


# Define predict function

@app.post('/predict')
def predict(age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope):
    data = pd.DataFrame([[age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope]])
    data.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    predictions = predict_model(model, data=data)
    print(data)
    return {'prediction': int(predictions['Label'][0])}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.1', port=8000)
