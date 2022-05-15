"""
The basis for this model is to look closer into the data. We haven't looked to close to the text we are training on and we may have made some mistakes on that the
(1) Multiple languages - similar Request may be hard to generalize when language is different. E.g. the model should understand translation of words.
(2) No quality check on tokens. There are multiple numbers in the vocabulary and misspelled words and weird abbreviations. The later two may be learned in the word embedding - but performance impact is unknown.
    It is how ever logical that if all texts has no spelling mistakes it would be easier to generalize.

(3) Initial run was on all categories with a 0.78 validation accuracy on epoch 5 and 6. From that point it went down again to the 0.75 area.
    This increase was not seen in the v7 model. I've seen these spikes before. It seems the tendency is to overfit right after this point.
(4) Try with just the ['Support', 'Request'] types and compare to the v7 run.
    This gave around 0.86 validation accuracy on the first epoch - following seems to overfit.
(5) Playing with a simpler model for v9

"""

# Remove duplicate rows in Item:
# DELETE FROM item WHERE id IN(SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) AS row_num);
import os
import re

import config
import pandas as pd

from bs4 import BeautifulSoup
from langdetect import detect_langs, detect
from logic.Prepare import Prepare
from logic.WordEmbedding import WordEmbedding

import numpy as np
import tensorflow as tf
from keras import layers, regularizers

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 200)
pd.options.mode.chained_assignment = None

table = 'request_tasktype_simple'

# First we get the data; we focus on a small subset.
p = Prepare(table=table, preprocess=False, limit=80000)

df = pd.read_csv(f'{config.BASE_PATH}/cache/{table}.csv')

if 'text' not in df.columns:
    df['text'] = df.apply(lambda x: BeautifulSoup(x['description'], 'html.parser').text, axis=1)
    df['text'] = df.apply(lambda x: x.text.replace(u'\u00A0', ' '), axis=1)
    df['text'] = df.apply(lambda x: re.sub('\n', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[\[\]]+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('\\\\', ' eller ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('bl\.a', ' blandt andet ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[0-9]+[\S]+[A-Øa-ø]+', ' qtmix ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[A-Øa-ø]+[\S]+[0-9]+', ' qtmix ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[0-9]+', ' qtnumber ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[^\s\dA-Øa-ø_]+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('\s\s+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(?<=mvh).+', '', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(?<=hilsen).+', '', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(?<=regards).+', '', x['text']), axis=1)
    df['text'] = df.apply(lambda x: x['text'].strip().lower(), axis=1)

    df = df[df.text != '']

    df['lang'] = df.apply(lambda x: detect(x['text']), axis=1)

    df = df[df.lang.isin(['da', 'no'])]
    df = df.to_csv(f'{config.BASE_PATH}/cache/{table}.csv')

w = WordEmbedding(table, index_text='text')

arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories \
    = p.fetch(
        amount=80000,
        index_label='tasktype',
        index_text='text',
        filter_categories=['Alerts', 'Incident', 'Procurement', 'Support for Students', 'Support for Students Incidents',
                           'Tasktype til Spam mail'])

embedding_dim = 128

vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

embeddings_index = {}
with open(f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs.txt') as f:
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
lx = layers.MaxPooling1D(5)(lx)
lx = layers.Conv1D(128, 5, activation="relu",
                   kernel_regularizer=regularizers.l2(l=0.01),
                   kernel_initializer='he_normal')(lx)
lx = layers.MaxPooling1D(5)(lx)
lx = layers.Dropout(0.4)(lx)
lx = layers.Conv1D(128, 5, activation="relu",
                   kernel_regularizer=regularizers.l2(l=0.01),
                   kernel_initializer='he_normal')(lx)
lx = layers.GlobalMaxPooling1D()(lx)
lx = layers.Dropout(0.4)(lx)
lx = layers.Dense(128, activation="relu")(lx)
lx = layers.Dropout(0.4)(lx)
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
model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val))

string_input = tf.keras.Input(shape=(1,), dtype="string")
v = vectorizer(string_input)
preds = model(v)
end_to_end_model = tf.keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["Brugeradministration: vi skal have genansat Per Børge Hansen på 01-02-2090"]]
)

print(categories[np.argmax(probabilities[0])])
