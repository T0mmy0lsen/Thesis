

import re

import config
import pandas as pd

from bs4 import BeautifulSoup
from langdetect import detect_langs, detect

from logic.GenerateForAssignee import GenerateForAssignee
from logic.Prepare import Prepare
from logic.Prepare_v2 import Prepare_v2
from logic.WordEmbedding import WordEmbedding

import numpy as np
import tensorflow as tf
from keras import layers, regularizers

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 200)
pd.options.mode.chained_assignment = None

"""
"""

table = 'request_tasktype_simple'

p = GenerateForAssignee()
df = p.fetch()

arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories \
    = p.fetch()

embedding_dim = 128

vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=100)
text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

embeddings_index = {}
we_path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{embedding_dim}.txt'
with open(we_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        if word == 'qtmix':
            print(word)
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
lx = layers.Conv1D(256, 5, activation="relu")(embedded_sequences)
lx = layers.GlobalAveragePooling1D()(lx)
lx = layers.Conv1D(256, 5, activation="relu")(embedded_sequences)
lx = layers.GlobalAveragePooling1D()(lx)
lx = layers.Dense(128, activation="relu")(lx)

preds = layers.Dense(len(categories), activation="softmax")(lx)
model = tf.keras.Model(int_sequences_input, preds)
model.summary()

x_train = vectorizer(np.array([[s] for s in arr_x_train])).numpy()
x_val = vectorizer(np.array([[s] for s in arr_x_validate])).numpy()

y_train = np.array(arr_y_train)
y_val = np.array(arr_y_validate)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)

model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))

string_input = tf.keras.Input(shape=(1,), dtype="string")
v = vectorizer(string_input)
preds = model(v)
end_to_end_model = tf.keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["Jeg skal bruge en ny computer"]]
)

print(categories[np.argmax(probabilities[0])])
print(categories[np.argmax(probabilities[1])])
print(categories[np.argmax(probabilities[2])])