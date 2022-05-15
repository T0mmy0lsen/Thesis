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
In the light of problems with labels, I'll start with looking into mean_squared_error:
https://towardsdatascience.com/report-time-execution-prediction-with-keras-and-tensorflow-8c9d9a889237

The result is:

mean_absolute_error: 0.8708
mean_squared_error: 1.2079
val_loss: 1.8391
val_mean_absolute_error: 1.0753
val_mean_squared_error: 1.8391

The process time range trained on is [0 - 7.7]

I was looking for insight for 'what is a good mean squared error (MSE) and found something on looking for outliers:
https://www.researchgate.net/post/What_is_the_Acceptable_MSE_value_and_Coefficient_of_determinationR2

My intuition tells me that since we used log() on the time it would have a reduced impact though - however, we still may have the problem of 'requests that timed out'. 
This refers to request that is completed or abandoned without closing the request at the time it happened. Say we have a request that is physically completed at 1, but closed at 4, because the involved simply forgot to close it.
Take the spam requests - these have 10% above the 1 mark, which all should have been closed in 1 or less, so we know there is errors in the data.

I should probably train some other models with mean squared error - but lets hope there is no real difference. 

Next step; save the model at do some manual testing. The goal is to see how bad it really is.
 -  This doesn't say much. If I use certain text, e.g. 'Automatisk svar: jeg er på ferie...' then it behaves as expected, but for the rest it just throws ~3.5 like a fit all processing time. 
    As I understand the model has a variance in the 1.0 area, which seems useless in practise. Simply because 'new' texts seems randomly assigned a average process time.
    
(1) First I thought it was a good idea with the log() time thing - but im afraid that the time (startDate - solutionDate) is just bad. 
    My thought is to try something like (assignedDate - lastChangeBeforeSolutionDate) but this will take som time to do this since I've to process all the request and inspect its relations, which takes a lot of time.
    I've noticed a timeConsumption column in the request, which may be helpful. 
    Right now I don't know what it measures, but it might be worth exploring if there is some correlation between the requests (i.e. the subject and description text) and the time consumption value.
(2) I've change the code so I can use all of the data I have. Before I filtered out if it was missing a startDate and solutionDate. It seems that the ones that have been filtered out is either spam or requests closed right away. 
    My best guess is that I've filtered out useful data for no reason. I'm setting the time consumption to 0 for all those who doesn't have a time consumption.
(3) I'm also retraining the IHLP17, since I limited it to 50.000 words which seems stupid when memory is not an issue and the word embedding is trained in 5-10 min. So why not make it better.

"""


class Model:

    def __init__(self):

        file = 'model_ihlp_sequential'

        p = PrepareTime(file)
        # w = WordEmbedding(file)

        arr_x_train, arr_y_train, arr_x_validate, arr_y_validate = p.fetch(80000, split=0.2, seed=1337, index_to_use='timeConsumption', index_filter=True)
        embedding_dim = 128

        vectorizer = layers.TextVectorization(max_tokens=100000, output_sequence_length=300)
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

        model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val))

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