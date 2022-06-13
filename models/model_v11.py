

import re

import config
import pandas as pd

from bs4 import BeautifulSoup
from langdetect import detect_langs, detect
from logic.Prepare import Prepare
from logic.Prepare_v2 import Prepare_v2
from logic.WordEmbedding import WordEmbedding

import numpy as np
import tensorflow as tf
from keras import layers, regularizers

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 200)
pd.options.mode.chained_assignment = None

"""
The idea is to that v11 will use more features from the data. We move the code from getting these feature out in the model. 
I've simplified the Prepare, thus making Prepare_v2.py

I've played around with some labels. I've defined solvedTime as the time from reactionTime to either last communication og solutionTime. What ever is first.
Since we use reactionTime its my best guess that workload is not important, since it states when Support got to that Request. 
Support may however open the Request, assign it and then go to the next Request.

- For solutionTime; with (x - min) / (max - min) the best MSE I've seen is 0.129. No overfitting.
- For reactionTime; with (x - min) / (max - min) the best MSE I've seen is 0.044. No overfitting. Its worth noting that 81% of Requests has a reactionTime less than 10 min.
- Low reaction can be text-dependent; e.g. text is spam, thus easy to close (no reactionTime)

To sum up for now:
- We can label Support and Request. This tells us whether the Request should be kept in the Servicedesk or outside.
- We can estimate reaction time and solution time quite fairly. Trained with data that is within 8 hours reactionTime and 1 week solutionTime, we have:
    - 21 min in error on average for reactionTime.
    - 309 min in error on average for solutionTime.
    
70483 has a reactionTime between 0 and 480
69582 has a reactionTime between 0 and 10
68966 has a reactionTime between 0 and 4
68237 has a reactionTime between 0 and 2
The average reactionTime is 1.45.
Using 1.45 for all gives a 1.38 min / (480/1.38) 0.0029 MAE for reactionTime.
    - On paper this is a better result, but we can't predict Requests that has long reactionTime. But do we need that?
    
57663 has a solvedTime between 0 and 2400
The average solvedTime is 288.
Using 288 for all gives a 358 min / (358/2400) = 0,149 MAE for solvedTime.

So, we have improvements on solvedTime, but reactionTime is bad. Probably no point in trying to predict that.

Creating a new word embedding with the communications, and subjects as well. I did a few nearest-k on some words and it made no sense. I probably made a bad word embedding.
One thing I've done is using a sentence length of 10 when learning the embedding - which might be harmful for the model.

First I'll just train the model after making the new embedding.
    - Results; 0.129 after 200 epochs. No Dropout was used. Train and validation accuracy are mostly the same.
    - So this was just run on the same word embedding. Don't know what I did wrong.
    
The new word embedding was made in WordEmbedding*-classes. The mayor issue here was the high amount of training data. Training the model (word embedding) as before just used all my 32GB ram and died out over night.
I've changed the data input to the model.fit() such that data is loaded in chunks. Moreover, the training data for the word embedding is outputted in ~500.000 lines each.

Right now I'm yielding 1024 each step (returning 1024 training rows from the data generator), which just seem like a waste of GPU utilization. 
But I'll let it run its 10 epochs and see if I get any good embeddings for words like 'database', 'bruger', 'password', which was absolutely useless before.

The CoNLL17 had some words that made sense compared to IHLP22. However, CoNLL17 came op with words that seems out of the Servicedesk context, which is probably why both word embeddings is performing the same.
Recall that GloVe did worse than both Danish word embeddings. This is most likely due to having known words with embeddings that can be trained on.
But those embeddings could be better, thus giving the rest of the deep learning model a hard time to learn anything useful.

Here it comes!

Say the word embeddings gives you a bunch of tags computed by frequency in Requests. These tags can be assigned to Supporters. The supporters takes a test that gives them the tags. The test can be made with prior Requests.
So if a Supporter can solve certain issues - they will be assigned task like them. Then Servicedesk becomes the expert system. Moreover, when some solves a task - they may be automatically strengthened by certain tags an even given new tags. 

What we do is assign Support issues to Supporters at work and schedule them to optimize throughput.
"""

timeConsumption = pd.read_excel(f'{config.BASE_PATH}/timeConsumption.xlsx')
# timeConsumption = pd.read_csv(f'{config.BASE_PATH}/request_timeconsumption.csv')
# timeConsumption = timeConsumption.fillna(0)

table = 'request_tasktype_simple'
# w = WordEmbedding(table)
p = Prepare_v2(table=table, limit=100)
df = p.fetch()
df = pd.merge(df, timeConsumption, how='inner')

df = df[df['reactionTime'] >= 0]
df = df[df['reactionTime'] < 480]
df = df[df['tasktype'] == 'Support']


def calc(x):
    # return x['reactionTime']
    return x['solvedTime'] - x['reactionTime']


def normalize(x, max_val):
    return x / max_val


df['label'] = df.apply(lambda x: calc(x), axis=1)
df = df[df['label'] < 480]
df = df[df['label'] >= 0]
max_val = df.label.max()
print(max_val)

df['labelNormalized'] = df.apply(lambda x: normalize(x['label'], max_val), axis=1)

print(df.head(5))

arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories \
    = p.ready(
        df=df,
        amount=80000,
        index_label='labelNormalized',
        index_text='text')

embedding_dim = 128

vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=250)
text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

embeddings_index = {}

# we_path = f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt'
# we_path = f'{config.BASE_PATH}/logic/input/GloVe/glove.6B.300d.txt'
# we_path = f'{config.BASE_PATH}/logic/input/IHLP17/vecs.txt'
we_path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{embedding_dim}.txt'

with open(we_path) as f:
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
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dropout(0.2)(x)

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

model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_val, y_val))

string_input = tf.keras.Input(shape=(1,), dtype="string")
v = vectorizer(string_input)
preds = model(v)
end_to_end_model = tf.keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["Jeg skal bruge en ny computer"]]
)

print(probabilities)
