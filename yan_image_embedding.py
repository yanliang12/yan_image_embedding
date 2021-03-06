###############yan_image_embedding.py##############

import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

'''
#Caused by: org.elasticsearch.index.mapper.MapperParsingException: The number of dimensions for field [logo_embedding] should be in the range [1, 2048] but was [4096]
'''

'''
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input 

base_model = NASNetMobile(
	weights="imagenet",
	)

model = Model(
	inputs = base_model.input, 
	outputs = base_model.get_layer('global_average_pooling2d_1').output)
'''

from tensorflow.keras.applications.densenet import *

densenet_base_model = DenseNet121()

densenet_embedding_model = Model(
	inputs = densenet_base_model.input, 
	outputs = densenet_base_model.get_layer('avg_pool').output)

def image_to_vector(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = densenet_embedding_model.predict(x)[0].tolist()
	return features
###############yan_image_embedding.py##############