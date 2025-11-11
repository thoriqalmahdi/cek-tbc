# feature_extractor_vgg16.py
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing import image

def build_vgg16_extractor():
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=x, name='vgg16_extractor')
    for layer in base.layers:
        layer.trainable = False
    return model

def extract_features(img_path, vgg_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_vgg = vgg_preprocess(np.copy(img_array))
    vgg_feat = vgg_model.predict(img_vgg, verbose=0).flatten()
    return vgg_feat
