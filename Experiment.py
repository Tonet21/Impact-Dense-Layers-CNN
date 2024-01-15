import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def extract_layer_weights(model, layer_name):
    layer = model.get_layer(layer_name)
    layer_weights = layer.get_weights()
    return layer_weights

def cosine_similarity_between_weights(weights1, weights2):
    flat_weights1 = np.concatenate([w.flatten() for w in weights1])
    flat_weights2 = np.concatenate([w.flatten() for w in weights2])
    min_len = min(len(flat_weights1), len(flat_weights2))
    flat_weights1 = flat_weights1[:min_len]
    flat_weights2 = flat_weights2[:min_len]
    flat_weights1 = flat_weights1.reshape(-1, 1)
    flat_weights2 = flat_weights2.reshape(-1, 1)
    similarity = cosine_similarity(flat_weights1.T, flat_weights2.T)
    return similarity[0, 0]


