from keras.models import load_model
from data.data import Dataset
from model.model import ModelFood

def testing():
    data = Dataset()
    model = load_model('model\food_classifier.keras', custom_objects={'ModelFood': ModelFood})
    loss, acc = model.evaluate(data.get_testing('x'), data.get_testing())
    print(f'Akurasi : {acc} | Loss : {loss}')
    
if __name__=='__main__':
    testing()