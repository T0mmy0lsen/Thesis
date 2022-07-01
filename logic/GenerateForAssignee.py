import os
import numpy as np
import config
import pandas as pd

from logic.GenerateText import GenerateText


class GenerateForAssignee:

    _df = pd.DataFrame()
    table = None

    def __init__(self):
        print('[Status] Preparing started')
        file_path_base = f'{config.BASE_PATH}/data'
        file_path_input = f'{config.BASE_PATH}/data/assignee.csv'

        if not os.path.isfile(file_path_input):
            
            item = pd.read_csv(f'{file_path_base}/item.csv', usecols=['id', 'username'])
            request = pd.read_csv(f'{file_path_base}/request.csv', usecols=['id', 'subject', 'description'])
            relation = pd.read_csv(f'{file_path_base}/relation.csv')

            item = item.fillna('')
            request = request.fillna('')
            relation = relation.fillna('')

            def addAssignees(x):
                tmp = relation[relation['leftId'] == x['id']]
                tmp = tmp[tmp['rightType'] == 'ItemRole']
                tmp = pd.merge(tmp, item, how='inner', left_on='rightId', right_on='id')
                lst = [e for e in list(tmp['username'].to_numpy()) if isinstance(e, str)]
                if len(lst) > 0:
                    return lst[:3][-1]
                return ''

            request['assignee'] = request.apply(lambda x: addAssignees(x), axis=1)
            request = GenerateText().add(request)

            request = request[['text', 'assignee']]
            request.to_csv(f'{file_path_base}/assignee.csv')

    def fetch(self, amount=77000, categorical=True):
        self._df = pd.read_csv(f'{config.BASE_PATH}/data/assignee.csv', usecols=['text', 'assignee'])
        self._df = self._df.fillna('')
        self._df = self._df[self._df['assignee'] != '']
        return self.ready(amount=amount, categorical=categorical)

    def ready(self,
              df=None,
              amount=1000,
              split=.25,
              seed=123,
              index_label='assignee',
              index_text='text',
              categorical=False):

        if df is not None:
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

GenerateForAssignee()