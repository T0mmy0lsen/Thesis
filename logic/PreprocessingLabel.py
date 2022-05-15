import datetime

import numpy as np
import pandas as pd


class PreprocessingLabel:

    def __init__(self, df=None):
        self.df = df

    def ihlp(self):
        print('[Status] Preprocessing label started')
        # self.df['processTime'] = self.df.apply(lambda x: self.get_process_time(x), axis=1)
        # self.df['processCategory'] = pd.qcut(self.df['processTime'], q=[0, .2, .4, .6, .8, 1], labels=[1, 2, 3, 4, 5])
        print('[Status] Preprocessing label ended')
        return self.df

    def get_process_time(self, x):
        try:
            received = datetime.datetime.strptime(str(x['receivedDate']), "%Y-%m-%d %H:%M:%S")
            solution = datetime.datetime.strptime(str(x['tblTimeStamp']), "%Y-%m-%d %H:%M:%S")
            x = int(solution.timestamp()) - int(received.timestamp())
            if x < 1:
                return 1
            return np.log(x) / np.log(10)
        except:
            return 1
