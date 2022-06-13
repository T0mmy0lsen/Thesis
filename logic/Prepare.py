import os
import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import config
import pandas as pd

from logic.PreprocessingLabel import PreprocessingLabel
from logic.PreprocessingText import PreprocessingText
from logic.WordEmbedding import WordEmbeddingLoader


class PrepareTime:

    p = None
    _df = None
    fetch_all = None
    table = None

    def __init__(self, table=''):
        self.table = table
        self.p = Prepare(table)

    def fetch(self, amount=1000, split=.2, seed=1337, index_to_use='timeConsumption', index_filter=None):

        def convert(x):
            if isinstance(x, str) and x == '':
                return 0
            return float(str(x))

        self._df = pd.read_csv(f'{config.BASE_PATH}/cache/{self.table}.csv')
        self._df = self._df.fillna('')

        if index_filter is not None:
            if index_to_use == 'timeConsumption':
                self._df[index_to_use] = self._df.apply(lambda x: convert(x[index_to_use]), axis=1)
                self._df = self._df[self._df[index_to_use] < 150]

        arr_x = self._df.head(amount)['processText'].to_numpy()
        arr_y = self._df.head(amount)[index_to_use].to_numpy()

        return self.p.shuffle_and_split(arr_x, arr_y, split, seed)


class Prepare:

    _df = pd.DataFrame()
    table = None

    def fetch(self,
              amount=1000,
              split=.2,
              seed=1337,
              filter_categories=None,
              index_label='processCategory',
              index_text='processText'):

        self._df = pd.read_csv(f'{config.BASE_PATH}/cache/{self.table}.csv')
        self._df = self._df.fillna('')
        
        categories = self._df.head(amount)[index_label].to_numpy()
        categories = np.sort(np.unique(categories))

        if filter_categories is not None:
            for el in filter_categories:
                self._df = self._df[self._df[index_label] != el]

        arr_x = self._df.head(amount)[index_text].to_numpy()
        arr_y = []

        if filter_categories is not None:
            categories = np.array([e for e in categories if e not in filter_categories])

        for idx, el in self._df.head(amount).iterrows():
            i = np.where(categories == el[index_label])
            arr_y.append(i)

        arr_y = np.array(arr_y).flatten()

        return Prepare.shuffle_and_split(arr_x, arr_y, split, seed, categories)

    def __init__(self, table=None, preprocess=True, limit=None):

        self.table = table

        # Main idea is to grab the data from the DB, pre-process text and labels and save them as a CSV.
        # This however only enables a single 'basic' preprocessing pipeline, therefore we may set preprocessing to False s.t. we handle that elsewhere.

        if table is not None:

            print('[Status] Preparing started')

            file_path_input = f'{config.BASE_PATH}/cache/{table}.csv'

            if not os.path.isfile(file_path_input):

                # Get all data or a limited set.
                if limit is not None:
                    # self._df = Data().get_limit(table, limit)
                    pass
                else:
                    # self._df = Data().get(table)
                    pass

                # Use default preprocessing pipeline
                if preprocess:
                    self._df = PreprocessingText(df=self._df).ihlp()
                    self._df = PreprocessingLabel(df=self._df).ihlp()

                self._df.to_csv(file_path_input, index=False)

            print('[Status] Preparing ended')

    @staticmethod
    def shuffle_and_split(arr_x, arr_y, split, seed, categories):

        # Shuffle the data
        seed = seed
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_x)
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_y)

        # Extract a training & validation split
        validation_split = split
        num_validation_samples = int(validation_split * len(arr_x))
        arr_x_train = arr_x[:-num_validation_samples]
        arr_x_validate = arr_x[-num_validation_samples:]
        arr_y_train = arr_y[:-num_validation_samples]
        arr_y_validate = arr_y[-num_validation_samples:]

        return arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories

    # Deprecated
    def fetch_as_word_embeddings(self, amount=None):

        # This data was pre-processed in the function build_as_word_embedding
        # The idea is to 'draw' the sentences as word_embedding vectors at the input.
        def build_as_word_embedding():

            self._df = pd.read_excel(f'{config.BASE_PATH}/cache/{self.table}.xlsx')

            # v = Value()
            w = WordEmbeddingLoader()
            max_length = 300

            for idx, el in self._df.iterrows():

                percent_processed = idx * 100 / len(self._df)
                sys.stdout.write("\r%f%% documents processed." % percent_processed)
                sys.stdout.flush()

                sentence_as_vector = np.empty((0, 1), float)
                sentence_as_string = el['processText']

                for index in range(0, max_length):

                    word = '[UNK]'

                    if isinstance(sentence_as_string, str):
                        explode = sentence_as_string.split(' ')
                        if len(explode) > index:
                            word = explode[index]

                    if word in w.word_embeddings_indexes:
                        word_embedding = w.word_embeddings[word]
                        sentence_as_vector = np.append(sentence_as_vector, np.array(word_embedding))
                    else:
                        sentence_as_vector = np.append(sentence_as_vector, np.array(w.word_embeddings['[UNK]']))

                sentence_as_vector_str = ' '.join([f'{e:.6f}' for e in sentence_as_vector])
                key = f'{self.table}'
                value = f'{self._df.iloc[idx].processCategory} {sentence_as_vector_str}'

                # v.set([0, key, value])

        # build_as_word_embedding()
        # v = Value(amount)

        print('[Status] Fetch started')
        # result = v.all(self.table)
        result = []
        print('[Status] Fetch ended')

        arr_x = []
        arr_y = []

        _input_x = np.empty((0, 300, 128), float)
        _input_y = np.empty((0, 1), int)

        for row in result:

            value = row[2].split(' ')
            arr_x.append(value[1:])
            arr_y.append(value[0])

            if len(arr_y) % 100 == 0:
                percent_processed = len(arr_y) * 100 / amount
                sys.stdout.write("\r%f%% rows processed." % percent_processed)
                sys.stdout.flush()

        print('')
        return np.array(arr_x).astype(float).reshape(amount, 300, 128), np.array(arr_y).astype(int)

    # Deprecated
    def fetch_as_tokens(self, amount):

        def create_and_train_tokenizer(texts):
            _tokenizer = Tokenizer()
            _tokenizer.fit_on_texts(texts)
            return _tokenizer

        def encode_requests(_tokenizer, _max_length, docs):
            encoded = _tokenizer.texts_to_sequences(docs)
            padded = pad_sequences(encoded, maxlen=_max_length, padding="post")
            return padded

        self._df = pd.read_excel(f'{config.BASE_PATH}/cache/{self.table}.xlsx')
        self._df = self._df.fillna('')

        arr_x = self._df.head(amount)['processText'].to_numpy()
        arr_y = []

        # Encode OneHot vector
        for idx, el in self._df.head(amount).iterrows():
            tmp_el = [0] * 5
            tmp_el[int(el['processCategory']) - 1] = 1
            arr_y.append(tmp_el)

        arr_y = np.array(arr_y)

        tokenizer = create_and_train_tokenizer(texts=arr_x)
        vocab_size = len(tokenizer.word_index) + 1
        print("Vocabulary size:", vocab_size)

        max_length = max([len(row.split()) for row in arr_x])
        print("Maximum length:", max_length)

        input_x = encode_requests(tokenizer, max_length, arr_x)
        input_y = arr_y

        return input_x, input_y, max_length, vocab_size, tokenizer