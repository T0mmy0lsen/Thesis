import numpy as np
from keras.constraints import unit_norm
from scipy import spatial

import config
from logic.Prepare import Prepare
from keras import layers, regularizers
import tensorflow as tf

from logic.WordEmbedding import WordEmbedding

"""
This time I'm stating from a working example. I took the example from https://keras.io/examples/nlp/pretrained_word_embeddings/. The code is found in /models/demo/keras_word_embedding.py.
The initial accuracy of the data from the example gave a validation accuracy of around 0.7 - which is not good, but it's using 20 categories, so it's not bad either.

If I switch to my data, the validation accuracy ends up around 0.38, however, it does peak around 0.44, so something is happening.
I've also switch to the CoNLL17 (danish word embedding) instead of the Glove used in the original example. The CoNLL17 show more frequent validation accuracies in the 0.44 area - so it seems it helps.
I've checked for missing word embeddings in CoNLL17 there are 5915 missing out of 14085, so that's a lot. 

The code can be found in /models/demo/keras_word_embedding_reworked.py

My next logical step would be: 

1. Try to train CoNLL17 with my data
2. (This file) Try the above but with the word embedding I've trained earlier.
3. (This file) Follow some steps from https://www.analyticsvidhya.com/blog/2020/09/overfitting-in-cnn-show-to-treat-overfitting-in-convolutional-neural-networks/

NOTICE: I've changed WordEmbedding to output the embedding in a single file instead of two. THis matches the Glove format.
The line w = WordEmbedding(file_name='word_embeddings') is set to build this new format.

(1) The run with my owned trained word embedding gave accuracy in the 0.40 range. Still over-fitting.
(2) Tried with regularization. 30 epochs hits accuracy of 0.64, but valuation accuracy is still in the 0.4 area.
(3) So far I've used 10.000 samples and 30 epochs. Will try with 100 epochs and 30.000 samples. Stay tuned.
(4) Result of (3): Still shitty. Pretty solid in the the 0.40 - 0.45 range, but nothing more. Training accuracy not over 0.6.
(5) As far as word embedding goes, the one I've trained is bad. If i search for 'sdu' I get usernames and other random numbers in the top 5.
    Next is the CoNLL17 this is missing a lot of words, so this might not be much better. With the (3) setup, CoNLL17 also keeps in the 0.4 - 0.45 area.
    
    Following is the top-5 for 'kode':
    
    IHLP17      -> ['operere', 'selvbetjeningsside', 'passwod', '20735070', 'nholg14']
    CoNLL17     -> ['koden', 'koder', 'mail-program', 'brugernavn/adgangskode', 'paypal.com']
    GloVe (100) -> ['sumantri', 'kubsch', 'basudev', 'katzmann', 'kalliope']
    
    On my data I get around 0.44 validation accuracy on all - so its hard to see any difference, when using the three word embeddings.
    Looking at the example (keras_word_embedding.py) I'm around 0.75 with the (3) setup.
    If I use CoNLL17 on the example data its clear that it suffers. Training accuracy gets a max on around 0.65.
    Thus, it is clear that the quality of the word embedding makes an impact regardless of the rest of the model.
    Moreover, the training accuracy may reach 0.65 even with a un-suitable word embedding.
    
    The conclusion must be. I need a better word embedding and the samples and labels I'm providing has a poor quality.
    I believe I have to dig deeper into the samples and labels to determine if labels are wrong (in any degree) and/or better text pre-processing is needed.
    
(6) Continue with the /models/analysis/text_preprocess*.py files for more. 
"""


class Model:

    def __init__(self):

        p = Prepare('model_ihlp_sequential')
        # w = WordEmbedding(file_name='word_embeddings')

        arr_x_train, arr_y_train, arr_x_validate, arr_y_validate = p.fetch(30000, split=0.2, seed=1337)
        class_names = [1, 2, 3, 4, 5]
        embedding_dim = 100

        vectorizer = layers.TextVectorization(max_tokens=50000, output_sequence_length=200)
        text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}
        with open(f'{config.BASE_PATH}/logic/input/GloVe/glove.6B.100d.txt') as f:
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
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal')(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu",
                          kernel_regularizer=regularizers.l2(l=0.01),
                          kernel_initializer='he_normal'
                          )(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

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