import os
this_dir = os.path.dirname(__file__)

from keras.models import model_from_json
import tensorflow as tf

def path(file):
	return os.path.join(this_dir, file)

def init():
  with open(path('model.json')) as data_file:
    loaded_model = model_from_json(data_file.read())
    loaded_model.load_weights(path('weights.h5'))
    print('Loaded model!')

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model