
import config

import numpy as np
import tensorflow as tf

from logic.Prepare import PrepareTime
from scipy import spatial


def find_closest_embeddings():

    def calc(embedding):
        return sorted(embeddings_index.keys(), key=lambda x: spatial.distance.euclidean(embeddings_index[x], embedding))

    embeddings_index = {}
    with open(f'{config.BASE_PATH}/logic/input/IHLP17/vecs.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    closest = calc(embeddings_index["kode"])[1:6]
    print(closest)

    embeddings_index = {}
    with open(f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    closest = calc(embeddings_index["kode"])[1:6]
    print(closest)

    embeddings_index = {}
    with open(f'{config.BASE_PATH}/logic/input/GloVe/glove.6B.100d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    closest = calc(embeddings_index["kode"])[1:6]
    print(closest)


def test_model():

    model_path = 'model_ihlp_sequential_v5.py_model'

    p = PrepareTime('model_ihlp_sequential')
    arr_x_train, arr_y_train, arr_x_validate, arr_y_validate = p.fetch(40000, split=0.2, seed=1337)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=50000, output_sequence_length=300)
    text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
    vectorizer.adapt(text_ds)

    loaded_model = tf.keras.models.load_model(f"{config.BASE_PATH}/saved_models/{model_path}")
    string_input = tf.keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = loaded_model(x)
    end_to_end_model = tf.keras.Model(string_input, preds)

    def predict_input():
        line = input('What is your request? \n')
        probabilities = end_to_end_model.predict(
            [[line]]
        )
        print(probabilities[0])
        predict_input()

    predict_input()