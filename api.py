from typing import List

from fastapi import FastAPI, Query
import tensorflow as tf
import numpy as np

app = FastAPI()


@app.get("/predict")
async def predict(values: List[float] = Query(None)):
  model = tf.keras.models.load_model('saved_model')
  prediction = model.predict(np.array([values]))
  output = np.argmax(prediction)
  to_return = "Persists" if output == 1 else "Not persists"
  return {"output": to_return}