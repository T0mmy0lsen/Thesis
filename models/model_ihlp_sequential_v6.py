import os

import numpy as np
from keras.constraints import unit_norm
from scipy import spatial

import config
from logic.Prepare import Prepare, PrepareTime
from keras import layers, regularizers
import tensorflow as tf

from logic.WordEmbedding import WordEmbedding

"""
At this point neither timeConsumption or the original processTime have worked. That is solutionTime - resolvedTime.
This time we assume that the real resolutionTime is that time of the last CommunicationSimple.
This is found by left joining the table Relation History and taking the latest type = CommunicationSimple for a Request.
This means we filter out around 40000, which has no communication steps, thus these would be assume instantly resolved.
This however seems unlikely, thus filtered from training.

(1) First round got 1.4 MSE, with 4.5 as average value. Retraining with a better word embedding. Last run had 20.000 misses out of 87.000.
(2) I'm expecting correlation between words and time. This should be true - take 'genansæt'. This mostly shows in standard e-mails that are sent to Servicedesk from CRM.
    The word is present in 10% of all requests and in general such requests has a timeConsumption in the 1 - 9 area (75%).
    I do however fear that there is no clear correlation between some similar tasks and the time consumption in most cases.
    To get better results we must analyse this - maybe by finding similar requests and see the distribution of time consumption.
(3) For now my time is probably better used on find skill-sets in the text rather than time estimation.
"""


class Model:

    def __init__(self):

        file = 'request_history_communication'

        p = PrepareTime(file)
        # w = WordEmbedding(file)

        arr_x_train, arr_y_train, arr_x_validate, arr_y_validate = p.fetch(46400, split=0.2, seed=1337, index_to_use='processTime', index_filter=True)
        print('Average: {}'.format(np.average(arr_y_train)))
        embedding_dim = 128

        vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=500)
        text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}
        with open(f'{config.BASE_PATH}/logic/input/IHLP17/vecs.txt') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
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
        x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D()(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="sigmoid")(x)

        preds = layers.Dense(1)(x)
        model = tf.keras.Model(int_sequences_input, preds)
        model.summary()

        x_train = vectorizer(np.array([[s] for s in arr_x_train])).numpy()
        x_val = vectorizer(np.array([[s] for s in arr_x_validate])).numpy()

        y_train = np.array(arr_y_train)
        y_val = np.array(arr_y_validate)

        optimizer = tf.keras.optimizers.SGD(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        model.fit(x_train, y_train, batch_size=32, epochs=400, validation_data=(x_val, y_val))

        # Save and try the model

        model.save(f"{config.BASE_PATH}/saved_models/{os.path.basename(__file__)}_model_v2")

        loaded_model = tf.keras.models.load_model(f"{config.BASE_PATH}/saved_models/{os.path.basename(__file__)}_model_v2")
        string_input = tf.keras.Input(shape=(1,), dtype="string")
        x = vectorizer(string_input)
        preds = loaded_model(x)
        end_to_end_model = tf.keras.Model(string_input, preds)

        probabilities = end_to_end_model.predict(
            [["Jeg kan ikke logge ind i Pure, kan i hjælpe mig?"]]
        )

        print(probabilities[0])