###############yan_image_embedding.py##############
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

base_model = VGG19(weights='imagenet')
model = Model(
	inputs=base_model.input, 
	outputs=base_model.get_layer('fc1').output)

def image_to_vector(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)[0].tolist()
	return features

###############yan_image_embedding.py##############