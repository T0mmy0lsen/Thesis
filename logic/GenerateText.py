import re
from bs4 import BeautifulSoup


class GenerateText:

    def __init__(self):
        pass

    def add(self, df):
        if 'text' not in df.columns:
            df['text'] = df.apply(lambda x: BeautifulSoup(x['subject'] + ". " + x['description'], 'html.parser').text, axis=1)
            df['text'] = df.apply(lambda x: x.text.replace(u'\u00A0', ' '), axis=1)
            df['text'] = df.apply(lambda x: re.sub('\n', ' ', x['text']), axis=1)
            df['text'] = df.apply(lambda x: re.sub('[\[\]]+', ' ', x['text']), axis=1)
            df['text'] = df.apply(lambda x: re.sub('\\\\', ' eller ', x['text']), axis=1)
            df['text'] = df.apply(lambda x: re.sub('[^\s\dA-Øa-ø_]+', ' ', x['text']), axis=1)
            df['text'] = df.apply(lambda x: re.sub('\s\s+', ' ', x['text']), axis=1)
            df['text'] = df.apply(lambda x: x['text'].strip().lower(), axis=1)
            df = df[df.text != '']

            # df['lang'] = df.apply(lambda x: detect(x['text']), axis=1)
            # df = df[df.lang.isin(['da', 'no'])]

            return df