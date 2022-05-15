import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import losses_utils

from logic.Prepare import Prepare
from keras import layers, models

"""
Not having to much success with previous tries, I've taken at step back and tried with tokenization, i.e. instead of a vector for each word, I've used a single integer for each word.
Since accuracy stayed around 0.38 I quickly abandoned the approach. Perhaps I'm making the same mistake with this model as the previous. 
I played around with batch size and dropout to prevent it from over-fitting, but at some point I got around 0.6 - 0.7 accuracy, but validation accuracy was still 0.38.
I'm stating to suspect that by categorization of the data is all wrong.

(1) Categories is 1 - 5, where 1 is lower solution-time.

"""


class Model:

    def __init__(self):

        p = Prepare('model_ihlp_sequential')
        input_x, input_y, max_length, vocab_size = p.fetch_as_tokens(10000)

        model = self.create_embedding_model(vocab_size, max_length)
        model.summary()

        self.fit(model, input_x, input_y, batch_size=100, epochs=100)

    def create_embedding_model(self, vocab_size, max_length):
        # Acc ~.38
        model = models.Sequential()
        model.add(layers.Embedding(vocab_size, 200, input_length=max_length))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.MaxPooling1D())
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(5, activation='sigmoid'))
        return model

    def fit(
            self,
            model,
            input_x,
            input_y,
            batch_size=None,
            epochs=10
    ):
        size = len(input_y)
        size_train = int(size * .8)

        input_x_test = input_x[size_train:]
        input_y_test = input_y[size_train:]

        input_x = input_x[:size_train]
        input_y = input_y[:size_train]

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(input_x, input_y, epochs=epochs, batch_size=batch_size)

        evaluate = model.evaluate(input_x_test, input_y_test, verbose=2)
        print(evaluate)
