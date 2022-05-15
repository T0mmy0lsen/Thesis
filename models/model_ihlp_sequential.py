import tensorflow as tf

from logic.Prepare import Prepare

"""
First approach was to encode the strings as their respective word embedding. The word embedding was trained on all 40.000 requests, see logic/WordEmbedding.py.py
I've used vector size 128, since it seemed a good estimate given most pretrained word embeddings comes in the size from 50 to 200.
Each sentence was reduced to a max length of 300. Text preprocessing was som basic cleanup; i.e. legitimatizing, remove stopwords and html-tag like removal with BeautifulSoup.
In all cases validation accuracy did not vary much from the .34 - .38 range, with the model over-fitting almost every time (cross-validation of test-data).

(1) Categories is 1 - 5, where 1 is lower solution-time.

"""


class Model:

    models = [
        # Acc ~.34
        tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(32, 8, activation="relu", input_shape=(300, 128)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ]),
        # Acc ~.34
        tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (32, 32), padding='same', activation="relu", input_shape=(300, 128, 1)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ]),
        # Acc ~.34
        tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(300, 128)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
    ]

    def __init__(self):
        self.fit(self.models[0], fetch_size=5000, epochs=20)

    def fit(
            self,
            model,
            fetch_size=100,
            batch_size=None,
            epochs=10
    ):

        p = Prepare('model_ihlp_sequential')
        input_x, input_y = p.fetch_as_word_embeddings(fetch_size)

        size = len(input_y)
        size_train = int(size * .8)

        input_x_test = input_x[size_train:]
        input_y_test = input_y[size_train:]

        input_x = input_x[:size_train]
        input_y = input_y[:size_train]

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        model.fit(input_x, input_y, epochs=epochs, batch_size=batch_size)

        evaluate = model.evaluate(input_x_test, input_y_test, verbose=2)
        print(evaluate)
