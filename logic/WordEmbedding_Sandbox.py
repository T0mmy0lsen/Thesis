import io
import os
import re
import sys

import tqdm
import config
import string

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers

# https://www.tensorflow.org/tutorials/text/word2vec


class Word2Vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        num_ns = 4
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns + 1)

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


class WordEmbedding:

    SEED = 1337
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, table='request_tasktype_simple', dim=128):

        path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{dim}.txt'
        file_path = f'{config.BASE_PATH}/cache/{table}_word_embedding.csv'

        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

        # Define the vocabulary size and the number of words in a sequence.
        vocab_size = 200000
        sequence_length = 100

        # Use the `TextVectorization` layer to normalize, split, and map strings to
        # integers. Set the `output_sequence_length` length to pad all samples to the
        # same length.
        vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=vocab_size + 1,
            output_mode='int',
            output_sequence_length=sequence_length)

        text_ds = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))
        vectorize_layer.adapt(text_ds.batch(1024))

        BATCH_SIZE = 512

        def tf_data_generator():
            files = [f'_{i}00000.dat' for i in range(1, 29)]
            files.append('_2907954.dat')
            for file in files:
                i = 0
                labels = np.genfromtxt(f'{config.BASE_PATH}/cache/word_embeddings/labels{file}', delimiter=',', dtype=np.int64)
                targets = np.genfromtxt(f'{config.BASE_PATH}/cache/word_embeddings/targets{file}', delimiter=',', dtype=np.int64)
                contexts = np.genfromtxt(f'{config.BASE_PATH}/cache/word_embeddings/contexts{file}', delimiter=',', dtype=np.int64)
                while i < len(labels) - BATCH_SIZE:
                    if i % BATCH_SIZE == 0:
                        yield (targets[i:i + BATCH_SIZE], contexts[i:i + BATCH_SIZE]), labels[i:i + BATCH_SIZE]
                    i = i + 1
                yield (targets[-BATCH_SIZE:], contexts[-BATCH_SIZE:]), labels[-BATCH_SIZE:]

        dataset = tf.data.Dataset.from_generator(tf_data_generator, args=[], output_signature=(
            (tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int64), tf.TensorSpec(shape=(BATCH_SIZE, 9), dtype=tf.int64)), tf.TensorSpec(shape=(BATCH_SIZE, 9), dtype=tf.int64)
        ))

        print(dataset)

        embedding_dim = dim
        word2vec = Word2Vec(vocab_size, embedding_dim)
        word2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        word2vec.fit(dataset, epochs=40)

        weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
        vocab = vectorize_layer.get_vocabulary()

        out_v = io.open(path, 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue
            vec = weights[index]
            out_v.write(f'{word} ' + ' '.join([str(x) for x in vec]) + "\n")
        out_v.close()

WordEmbedding()