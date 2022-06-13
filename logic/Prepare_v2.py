import os
import numpy as np
import config
import pandas as pd

"""
How to use:
1. Create a view in the database that suits your need.
2. Call p = Prepare_v2({table}) this saves the database as a CSV. This is done since I usually create a view which may or may not have bad performance
3. Call p.fetch() to load the CSV to a DataFrame.
4. Do what ever you want with the DataFrame, just make sure to have a column with the input data and another with corresponding labels.
5. Call p.ready() to create appropriate training and testing sets.
"""


class Prepare_v2:

    _df = pd.DataFrame()
    table = None

    def __init__(self, table=None, limit=None):

        self.table = table

        # Main idea is to grab the data from the DB, pre-process text and labels and save them as a CSV.
        # This however only enables a single 'basic' preprocessing pipeline, therefore we may set preprocessing to False s.t. we handle that elsewhere.

        if table is not None:

            print('[Status] Preparing started')

            file_path_input = f'{config.BASE_PATH}/cache/{table}.csv'

            if not os.path.isfile(file_path_input):

                # Get all data or a limited set.
                """ Get the data by export from DB """
                if limit is not None:
                    # self._df = Data().get_limit(table, limit)
                    pass
                else:
                    # self._df = Data().get(table)
                    pass

                self._df.to_csv(file_path_input, index=False)

            print('[Status] Preparing ended')

    def fetch(self):
        self._df = pd.read_csv(f'{config.BASE_PATH}/cache/{self.table}.csv')
        self._df = self._df.fillna('')
        return self._df

    def ready(self,
              df=None,
              amount=1000,
              split=.1,
              seed=123,
              index_label='label',
              index_text='text',
              categorical=False):

        self._df = df

        categories = []
        if categorical:
            categories = self._df.head(amount)[index_label].to_numpy()
            categories = np.sort(np.unique(categories))

        arr_x = self._df.head(amount)[index_text].to_numpy()
        arr_y = []

        if categorical:
            for idx, el in self._df.head(amount).iterrows():
                i = np.where(categories == el[index_label])
                arr_y.append(i)
        else:
            arr_y = self._df.head(amount)[index_label].to_numpy()

        arr_y = np.array(arr_y).flatten()

        return self.shuffle_and_split(arr_x, arr_y, split, seed, categories)

    def shuffle_and_split(self, arr_x, arr_y, split, seed, categories):

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