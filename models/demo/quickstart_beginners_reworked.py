import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from logic.Prepare import Prepare
from logic.WordEmbedding import WordEmbeddingLoader

INPUT = 'input/quickstart_beginners_reworked/'


def build():

    w = WordEmbeddingLoader()
    df = Prepare(file_name='quickstart_beginners', fetch_all=False).df

    max_length = 300
    _input_x = np.empty((0, max_length, w.word_embeddings_dimensions), float)
    _input_y = np.empty((0, 1), int)

    for idx, el in df.iterrows():

        sentence_as_vector = np.empty((0, w.word_embeddings_dimensions), float)
        sentence_as_string = el['processText']

        percent_processed = idx * 100 / len(df)
        sys.stdout.write("\r%f%% documents processed." % percent_processed)
        sys.stdout.flush()

        for index in range(0, max_length):

            explode = sentence_as_string.split(' ')
            word = '[UNK]'

            if len(explode) > index:
                word = explode[index]

            if word in w.word_embeddings_indexes:
                word_embedding = w.word_embeddings[word]
                sentence_as_vector = np.append(sentence_as_vector, np.array([word_embedding]), axis=0)
            else:
                sentence_as_vector = np.append(sentence_as_vector, np.array([w.word_embeddings['[UNK]']]), axis=0)

        _input_x = np.append(_input_x, [sentence_as_vector], axis=0)
        _input_y = np.append(_input_y, df.iloc[idx].processCategory)

    np.savetxt(f'{INPUT}/shape.txt', _input_x.shape, fmt='%i')
    np.savetxt(f'{INPUT}/input_x.txt', _input_x.reshape(-1), fmt='%f')
    np.savetxt(f'{INPUT}/input_y.txt', _input_y, fmt='%i')

if not os.path.exists(f'{INPUT}input_x.txt'):
    build()

shape = pd.read_csv(f'{INPUT}/shape.txt', header=None).to_numpy().reshape(-1)
input_x = pd.read_csv(f'{INPUT}/input_x.txt', header=None).to_numpy().reshape(shape)
input_y = pd.read_csv(f'{INPUT}/input_y.txt', header=None).to_numpy().reshape(-1)

input_x_test = input_x[275:]
input_y_test = input_y[275:]

input_x = input_x[:275]
input_y = input_y[:275]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(300, 128)),
  tf.keras.layers.Dense(300, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(25)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(input_x, input_y, epochs=5)
evaluate = model.evaluate(input_x_test,  input_y_test, verbose=2)
print(evaluate)