

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

"""
(1) For fun, lets try other word embeddings; max 25 epochs, 128 batch size.

    GlobalAveragePooling1D()
    Dense(128, activation="relu")
    
    - IHLP22: 0.857
    - IHLP17: 0.855
    - GloVe (300): 0.836
    - CoNLL17: 0.852
    
    GloVe did not perform as well as the others. But the rest is hard to make any conclusions from.
    Lets throw in an extra layer - maybe there is not enough trainable parameters. I mean - it doesn't overfit at the moment, so it may underfit.
    
    Conv1D(32, 5, activation="relu")
    GlobalAveragePooling1D()
    Dense(128, activation="relu")

    - CoNLL17: 0.868
    
    (1.1) Clear improvement; give it some more params to play with
    
    Conv1D(32, 5, activation="relu")
    GlobalAveragePooling1D()
    Dense(256, activation="relu")
    
    - CoNLL17: 0.869
    
    (1.2) Small to no improvement; more kernels, less params in the dense layer
    
    Conv1D(64, 5, activation="relu")
    GlobalAveragePooling1D()
    Dense(128, activation="relu")
    
    - CoNLL17: 0.871
    - IHLP22: 0.870
    
    (1.3) Small to no improvement; more kernels, remove the dense layer
    
    Conv1D(128, 5, activation="relu")
    GlobalAveragePooling1D()
    
    - CoNLL17: 0.867
    
    (1.4) Not any better; so better keep the later.
    
(2) I've increased the input to 500 which just made everything overfit a lot faster. 0.95 accuracy on epoch 15.
    This seems like the generalization was mostly due to truncation of the text.
(3) Lets use some more dimensions on the Word Embedding.
    - IHLP22: 0.869 on 256 dimensions.
(4) Thoughts so far:
    - I've used to much time on the time estimation part. The labels had questionable labeling - which I should have realised earlier. I used to much time to conclude it from bad accuracy on my models.
      I did however make it quite clear that time needed for tasks was overlapping (high variance for same type of tasks) which lead to higher accuracy if I only trained on request that had either short or long solution-times.
    - Like I've come to realize in project management - we have to understand the process - without that we can't make a solution.
    - Next, if I make something I've to be sure that someone is going to use it. This and the prior can be done if I get Bilal to help.
    - Some parts I already know, regarding the process. I can take a request a see the lifecycle from which I can descripe a process.
        (1) I can argue that the more labels the more complex the problems becomes. So if remove som labels it should be easier. 
            From the categories I see '*Student*'. Now theses request should go somewhere else, since Servicedesk is not for students. I can argue that these are simply outliers.
            Next is spam, this should be removed as well. In practice I'd use spam detection. Probably train a model using transfer learning. Should be fairly easy to do.
            For now I focus on the remaining labels. Theses labels, say Support says a lot about where the work goes. Support usually stays in Servicedesk and may be handled by many. 
            The solution-time may be highly dependant on the current workload in Servicedesk as well as general causes such as holiday, complexity (hard to to, but we have specialist for them) and 
            hard-to-place requests (these may have no obvious solution or person to solve them).
            Best result: 0.870 (only Support and Request)
            Best result: 0.837 (without student and spam)
            Best result: 0.826 (without student)
            Best result: 0.791
        (2) I could just change e.g. 'Support for Students' to 'Support'. Then I'd know if it where requested by a student-mail.
        (3) Priority is usually kept on 3, so there is never given any real priority other than Incidents being more important than Request / Support
    - Can we use the same idea with word embeddings with the Solvers?
        (1) So a Solver (a person being part of the solution to a Request) would have a feature vector. 
            The feature vector can be trained on:
                - The text of the Request, this can be represented as we do now
                - Solution time spent, from when the Request was given until either given to another, a communication step or solution is given.
                  This should be analysed how to do.  
                - Workload on given time. Its hard to predict a certain Solvers workload - but it may be helpful to look at how many open Requests is assigned to the department or 
                  a certain skill-group (like many Requests are open with a Tag of Database).
     
"""

# We expect to have run v8 for the initial csv-file and word embedding
table = 'request_tasktype_simple'
p = Prepare(table=table, preprocess=False, limit=80000)
w = WordEmbedding(table, index_text='text', dim=128)

# filter_categories = ['Support for Students', 'Support for Students Incidents']
# filter_categories = ['Support for Students', 'Support for Students Incidents', 'Tasktype til Spam mail']
filter_categories = ['Alerts', 'Incident', 'Procurement', 'Support for Students', 'Support for Students Incidents', 'Tasktype til Spam mail']

arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories \
    = p.fetch(
        amount=80000,
        index_label='tasktype',
        index_text='text',
        filter_categories=filter_categories)

embedding_dim = 256

vectorizer = layers.TextVectorization(max_tokens=200000, output_sequence_length=100)
text_ds = tf.data.Dataset.from_tensor_slices(arr_x_train).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

embeddings_index = {}
# we_path = f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt'
# we_path = f'{config.BASE_PATH}/logic/input/GloVe/glove.6B.300d.txt'
we_path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{embedding_dim}.txt'
# we_path = f'{config.BASE_PATH}/logic/input/IHLP17/vecs.txt'
with open(we_path) as f:
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

lx = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
lx = layers.GlobalAveragePooling1D()(lx)
lx = layers.Dense(64, activation="relu")(lx)

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

model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))

string_input = tf.keras.Input(shape=(1,), dtype="string")
v = vectorizer(string_input)
preds = model(v)
end_to_end_model = tf.keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["Jeg skal bruge en ny computer"]]
)

print(categories[np.argmax(probabilities[0])])
