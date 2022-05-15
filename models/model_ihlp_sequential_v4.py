import numpy as np
from keras.constraints import unit_norm
from scipy import spatial

import config
from logic.Prepare import Prepare
from keras import layers, regularizers
import tensorflow as tf

from logic.WordEmbedding import WordEmbedding

"""
Thoughts on the processTime labels:

Around 2.5% is spam and 90% of that has the lowest process time, thus process time on that seems correct. This may be because the dispatcher can close the issue right away.

We are taking log() time, so a category 5 could easily be holding 'timeout' request. The other day I myself closed several request that should have been closed long ago.
E.g. I've sent the solution to the requester, but the requester did not respond or close the request - so the time would be completely of.
One way we could look at this was removing the time between the last activity and the time the issue was closed. One could ague that closing the request is not what solves it, but actually the last step in the request.
Take the spam as an example. The is no step in that - its actually solved the second its received by the system.

To test this - we remove all processTime category '5'. The idea is - the longer the process time, the more likely it is to be miss-classified.

So far no differences. The prior setup with 5 categories gave roughly 2/5 accuracy, i.e. twice the accuracy as a random guess. 
So, I'd expect this (without category 4 and 5) to do better than 0.6 if there was any difference. Which it did not.

I've tried both CoNLL17 and IHLP17 - no real difference here.

It seems to keep over-fitting threw in som more drop-out and tried longer samples length - still not change.It just over-fits slower.

(1) I've tried removing category 2 and 4, which gave an accuracy of 0.74 which is noticeably better than with removing 4 and 5.
    We are trying to predict time estimates, so we should probably look into some other loss-function and evaluation, since if truth is 5, then 4 is certainly better than 1, which we dont capture. 
    Categorically we may have overlap in 1 and 2, 2 and 3, but less likely 1 and 3, thus removing category 2 would make it easier to distinguish between them. This makes good sense.
(2) Removing 2, 3 and 4 gives us a very nice performance on 0.93 in validation accuracy. So its very noticeable that the time categories are very blurry. 
    The next step would be better categorization - this surely does a big difference.
(3) To create some intuition about what the word embedding does I've tried with the GloVe which gave an accuracy around 0.91. So the IHLP17 does help a bit.
    However, I'm certain that there is improvements to find with a better trained word embedding - but for know the focus should be on the quality of the labels.
"""


class Model:

    def __init__(self):

        p = Prepare('model_ihlp_sequential')
        # w = WordEmbedding(file_name='word_embeddings')

        filter_categories = [2, 4]
        arr_x_train, arr_y_train, arr_x_validate, arr_y_validate = p.fetch(80000, split=0.2, seed=1337, filter_categories=filter_categories)
        class_names = [e for e in [1, 2, 3, 4, 5] if e not in filter_categories]
        embedding_dim = 128

        vectorizer = layers.TextVectorization(max_tokens=50000, output_sequence_length=300)
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
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_initializer='he_normal')(embedded_sequences)
        x = layers.MaxPooling1D()(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal'
                          )(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.6)(x)

        preds = layers.Dense(len(class_names), activation="softmax")(x)
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
            [["this message is about computer graphics and 3D modeling"]]
        )

        print(class_names[np.argmax(probabilities[0])])