
import warnings
import pandas as pd

from objs.Request import Request

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# Lemmatizer
# Tokenizer
# Remove stopwords
# MongoDB for text search
# https://github.com/kazimirzak/Bachelor/blob/b3c5441ccb46d100b9eb8632a47c69b08761df90/main.py#L96
# https://jovian.ai/diardanoraihan/ensemble-cr/v/2?utm_source=embed#C39
# https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb


class Data:

    def get(self, table):
        rs = Request().get(table)
        data = pd.DataFrame(rs, columns=Request().fillables[table])
        data = data.fillna('')
        return data

    def get_limit(self, table, limit):
        rs = Request().get_limit(table, limit)
        data = pd.DataFrame(rs, columns=Request().fillables[table])
        data = data.fillna('')
        return data
