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


class WordEmbeddingLoaderGenism:

    word_embeddings = None
    word_embeddings_indexes = None
    word_embeddings_dimensions = 0

    def __init__(self):

        print('[Status] WordEmbeddingLoaderGenism started')

        path_embeddings = f'{config.BASE_PATH}/logic/input/CoNLL17/vecs.txt'
        df_embeddings = pd.read_csv(path_embeddings, sep=' ', header=None, skiprows=1, encoding='utf-8')

        word_embeddings = {}

        for idx, el in df_embeddings.iterrows():
            word_embeddings[el[0]] = el.to_numpy()[1:-1]
            if idx % 100 == 0:
                percent_processed = idx * 100 / len(df_embeddings)
                sys.stdout.write("\r%f%% documents processed." % percent_processed)
                sys.stdout.flush()

        self.word_embeddings_dimensions = len(df_embeddings.columns) - 1
        self.word_embeddings_indexes = df_embeddings[0].to_numpy()
        self.word_embeddings = word_embeddings

        print('[Status] WordEmbeddingLoader ended')


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
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


class WordEmbeddingLoader:

    word_embeddings = None
    word_embeddings_indexes = None
    word_embeddings_dimensions = 0

    def __init__(self):

        print('[Status] WordEmbeddingLoader started')

        path_words = f'{config.BASE_PATH}/logic/input/metadata.tsv'
        path_vectors = f'{config.BASE_PATH}/logic/input/vectors.tsv'

        df_words = pd.read_csv(path_words, sep='\t', header=None)
        df_vectors = pd.read_csv(path_vectors, sep='\t', header=None, dtype=float)

        word_embeddings = {}

        for idx, el in df_words.iterrows():
            word_embeddings[el[0]] = df_vectors.iloc[idx].to_numpy().flatten()

        self.word_embeddings_dimensions = len(df_vectors.columns)
        self.word_embeddings_indexes = df_words[0].to_numpy()
        self.word_embeddings = word_embeddings

        print('[Status] WordEmbeddingLoader ended')


class WordEmbedding:

    SEED = 1337
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, table='request', index_text='text', dim=64):

        path = f'{config.BASE_PATH}/logic/input/IHLP22/{table}_vecs_{dim}.txt'
        if os.path.exists(path):
            msg = f'Can not override existing word embedding {path}'
            print(msg)
            return

        file_path = f'{config.BASE_PATH}/cache/{table}_word_embedding.csv'

        if not os.path.exists(file_path):
            df = pd.read_csv(f'{config.BASE_PATH}/cache/{table}.csv')
            df = df.fillna('')
            df = df[[index_text]]
            df.to_csv(file_path, header=False, index=False, encoding='utf-8')

        text_ds = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        # Now, create a custom standardization function to lowercase the text and
        # remove punctuation.
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

        vectorize_layer.adapt(text_ds.batch(1024))

        # Vectorize the data in text_ds.
        text_vector_ds = text_ds.batch(1024).prefetch(WordEmbedding.AUTOTUNE).map(vectorize_layer).unbatch()

        sequences = list(text_vector_ds.as_numpy_iterator())  # It gets slow from this point

        targets, contexts, labels = self.generate_training_data(
            sequences=sequences,
            window_size=4,
            num_ns=8,
            vocab_size=vocab_size,
            seed=WordEmbedding.SEED)

        exit(0)

        targets = np.array(targets)
        contexts = np.array(contexts)[:, :, 0]
        labels = np.array(labels)

        print('\n')
        print(f"targets.shape: {targets.shape}")
        print(f"contexts.shape: {contexts.shape}")
        print(f"labels.shape: {labels.shape}")

        BATCH_SIZE = 1024
        BUFFER_SIZE = 10000
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        print(dataset)

        dataset = dataset.cache().prefetch(buffer_size=WordEmbedding.AUTOTUNE)
        print(dataset)

        embedding_dim = dim
        word2vec = Word2Vec(vocab_size, embedding_dim)
        word2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        word2vec.fit(dataset, epochs=200, callbacks=[tensorboard_callback])

        weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
        vocab = vectorize_layer.get_vocabulary()

        out_v = io.open(path, 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue
            vec = weights[index]
            out_v.write(f'{word} ' + ' '.join([str(x) for x in vec]) + "\n")
        out_v.close()

    def write_out(self, targets, contexts, labels, idx):

        # Targets
        str_data = "\n".join([str(x) for x in targets])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/cache/word_embeddings/targets_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

        # Context
        tmp = [e.numpy().flatten() for e in contexts]
        str_data = "\n".join([", ".join([str(y) for y in x]) for x in tmp])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/cache/word_embeddings/contexts_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

        # Labels
        tmp = [e.numpy().flatten() for e in labels]
        str_data = "\n".join([", ".join([str(y) for y in x]) for x in tmp])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/cache/word_embeddings/labels_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.

    def generate_training_data(self, sequences, window_size, num_ns, vocab_size, seed):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []
        idx = 0

        # Build the sampling table for `vocab_size` tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        # Iterate over all sequences (sentences) in the dataset.
        for sequence in tqdm.tqdm(sequences):

            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=WordEmbedding.SEED,
                    name="negative_sampling")

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1)

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

            if idx % 100000 == 0:
                self.write_out(targets, contexts, labels, idx)
                targets, contexts, labels = [], [], []
            idx = idx + 1

        self.write_out(targets, contexts, labels, idx)
        return targets, contexts, labels
