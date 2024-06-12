import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import Augmentor

class Dataset:
  def __init__(self, root_folder="/content/food-cl"):
    self.X = list()
    self.Y = list()

    for label in glob.glob(root_folder+"/*"):
      for img_path in glob.glob(label+"/*"):
        img = self.preprocessing(img_path)
        self.X.append(img)
        self.Y.append(label.split('/')[-1])

    self.Y_encoded = LabelEncoder().fit_transform(self.Y)
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(self.X), np.array(self.Y_encoded), 
                                                                            random_state=42, test_size=0.3)

  def preprocessing(self, img):
    img = cv2.imread(img)
    img = cv2.resize(img, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255    
    return img

  def get_training(self, fitur='y'):
    if fitur == 'x':
      return self.x_train
    else:
      return self.y_train

  def get_testing(self, fitur='y'):
    if fitur == 'x':
      return self.x_test
    else:
      return self.y_test


class AugmentorDataset:
  def __init__(self, path, size):
    self.path = path
    self.size = size

  def create_data_augmentation(self):
    for label_path in glob.glob(self.path+'/*'):
      pipeline = Augmentor.Pipeline(source_directory=label_path, output_directory='.')
      pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=20)
      pipeline.flip_left_right(probability=0.5)
      pipeline.flip_top_bottom(probability=0.5)
      pipeline.random_distortion(probability=0.8, grid_width=4, grid_height=4, magnitude=8)
      pipeline.shear(probability=0.6, max_shear_left=20, max_shear_right=20)
      pipeline.crop_centre(probability=0.2, percentage_area=0.6)
      pipeline.sample(self.size)