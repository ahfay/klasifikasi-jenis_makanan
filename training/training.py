from keras.callbacks import EarlyStopping
from model.model import ModelFood
from data.data import Dataset, AugmentorDataset

def training():
    root_dir = "/content/food-cl"
    augmentor = AugmentorDataset(root_dir, 500)
    augmentor.create_data_augmentation()
    data = Dataset()
    model = ModelFood()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data.get_training('x'), data.get_training(), epochs=25, 
                        validation_split=0.1, batch_size=32)
    model.save('model/food_classifier.keras')
    
    
if __name__=='__main__':
    training()
