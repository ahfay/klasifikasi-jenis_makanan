import numpy as np
from data.data import Dataset
from model.model import ModelFood
from keras.models import load_model

label = {0:'dessert', 
         1:'drink', 
         2:'meal'}

data = Dataset()
img = '/content/food-cl/meal/IMG_4297.jpg'
img = data.preprocessing(img)
img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch

model = load_model('model/food_classifier.keras', custom_objects={'ModelFood': ModelFood})
pred = np.argmax(model.predict(img), axis=1)
print(label.get(pred[0]))