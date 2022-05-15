import numpy as np
from keras.constraints import unit_norm
from scipy import spatial

import config
from logic.Prepare import Prepare
from keras import layers, regularizers
import tensorflow as tf

from logic.WordEmbedding import WordEmbedding

"""

Initial is 0.75 validation accuracy.

32378   41%     1935805	Request
27333   34%     1935807	Support
8910    11%     1935809	Support for Students
6708    8%      1935811	Procurement
3572    4%      4644243	Tasktype til Spam mail
78901   98%     Total

Only using Request and Support, 0.85 validation accuracy.

I'd hope for better results on the categorical training with the Request and Support only. These are labeled by the supporters, so it can't get better than that. 
Sure they may labeled them 'wrong' but we can't getter better labels than that.

So correlations between label and text is hard. Maybe this is caused by bad quality text? But why can a supporter label it?
- The supporter understands the original text better? Who could this be?
- The original text losses meaning after pre-processing? How could this be?

"""


class Model:

    def __init__(self):

        file_name = 'request_tasktype'

        p = Prepare(file_name)
        # w = WordEmbedding(file_name)

        arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories \
            = p.fetch(80000,
                      split=0.2,
                      seed=1337,
                      category_index='tasktype',
                      filter_categories=['Alerts', 'Incident', 'Procurement', 'Support for Students', 'Support for Students Incidents', 'Tasktype til Spam mail']
            )

        embedding_dim = 100

        vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=200)
        text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}
        with open(f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt') as f:
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
        x = layers.Conv1D(256, 5, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal')(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        preds = layers.Dense(len(categories), activation="softmax")(x)
        model = tf.keras.Model(int_sequences_input, preds)
        model.summary()

        x_train = vectorizer(np.array([[s] for s in arr_x_train])).numpy()
        x_val = vectorizer(np.array([[s] for s in arr_x_validate])).numpy()

        y_train = np.array(arr_y_train)
        y_val = np.array(arr_y_validate)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
        )
        model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val))

        string_input = tf.keras.Input(shape=(1,), dtype="string")
        x = vectorizer(string_input)
        preds = model(x)
        end_to_end_model = tf.keras.Model(string_input, preds)

        probabilities = end_to_end_model.predict(
            [["Brugeradministration: vi skal have genansat Per Børge Hansen på 01-02-2090"]]
        )

        print(categories[np.argmax(probabilities[0])])