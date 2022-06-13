import numpy as np
import pandas as pd
import config
import tensorflow as tf


def load_word_embedding():

    table = 'request_tasktype_simple'
    embedding_dim = 128
    embeddings_index = {}
    file_path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{embedding_dim}.txt'

    with open(file_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embedding_dim


def similarity():

    def run(path, word, dim=64):

        # https://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding

        x = 0
        embeddings_index = {}
        embedding_dim = dim
        table = 'request_tasktype_simple'

        # we_path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{embedding_dim}.txt'
        # we_path = f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt'
        with open(path) as f:
            for line in f:
                w, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[w] = coefs

        data = embeddings_index.items()
        data_list = list(data)
        embedding_words = [e[0] for e in data_list if len(e[1]) == embedding_dim]
        embedding = [e[1] for e in data_list if len(e[1]) == embedding_dim]

        index = embedding_words.index(word)

        embedding = np.array(embedding)
        batch_array = np.array([embedding[index]])

        normed_embedding = tf.nn.l2_normalize(embedding, dim=1)
        normed_array = tf.nn.l2_normalize(batch_array, dim=1)

        cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))
        closest_k_words = tf.nn.top_k(cosine_similarity, k=10)
        closest_k_words_arr = closest_k_words[1].numpy()[0]

        print([item for (idx, item) in enumerate(embedding_words) if idx in closest_k_words_arr])

    dim = 128
    table = 'request_tasktype_simple'
    path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{dim}.txt'
    run(path, 'kode', dim)


def word_occurrences():

    def count_occurrences(df, words_index):
        series = df[words_index].str.split(expand=True).stack().value_counts()
        series.sort_values(ascending=False)
        return series.to_dict()

    table = 'request_tasktype_simple'
    file_path = f'{config.BASE_PATH}/cache/{table}.csv'
    df = pd.read_csv(file_path, usecols=['text'])
    count = count_occurrences(df, 'text')
    with open('count.csv', 'w') as f:
        for key in count.keys():
            f.write("%s, %s\n" % (key, count[key]))

# This does not have the right data to run.
"""
def model():

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
"""
