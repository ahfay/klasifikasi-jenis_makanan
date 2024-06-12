from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.saving import register_keras_serializable

@register_keras_serializable(package='Custom', name='ModelFood')
class ModelFood(Model):
  def __init__(self):
    super(ModelFood, self).__init__()
    self.conv1 = Conv2D(16,(3,3),input_shape=(200,200,3),activation='relu')
    self.conv2 = Conv2D(8,(3,3),activation='relu')
    self.conv3 = Conv2D(4,(3,3),activation='relu')
    self.maxpool = MaxPooling2D(2,2)
    self.flatten = Flatten()
    self.fc1 = Dense(60,activation='relu')
    self.fc2 = Dense(30,activation='relu')
    self.fc3 = Dense(3,activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
