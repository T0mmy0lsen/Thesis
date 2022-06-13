import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import config


def text():
    print('Loading and merging text...')

    table_request = 'request_tasktype_simple'

    df_request = pd.read_csv(f'{config.BASE_PATH}/cache/{table_request}.csv')
    # df_request = df_request.head(10)

    df_request_text = df_request[['description']]
    df_request_text = df_request_text.rename(columns={'description': 'text'})

    df_request_subject = df_request[['subject']]
    df_request_subject = df_request_subject.rename(columns={'subject': 'text'})

    table_communication = 'communication'

    df_communication = pd.read_csv(f'{config.BASE_PATH}/cache/{table_communication}.csv')
    # df_communication = df_communication.head(10)

    df_communication_text = df_communication[['message']]
    df_communication_text = df_communication_text.rename(columns={'message': 'text'})

    df_communication_subject = df_communication[['subject']]
    df_communication_subject = df_communication_subject.rename(columns={'subject': 'text'})

    df = pd.concat([df_request_text, df_request_subject, df_communication_text, df_communication_subject],
                   ignore_index=True)

    df = df.fillna('')
    df_sentences = pd.DataFrame([], columns=['text'])

    print('Transforming text...')

    df['text'] = df.apply(lambda x: BeautifulSoup(x['text'], 'html.parser').text, axis=1)
    df['text'] = df.apply(lambda x: re.sub('(bl\.a\.|bl\.a)', ' blandt andet ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(evt\.)', ' eventuelt ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[?!.]', '\n', x['text']), axis=1)

    print('Constructing a lot of lines...')

    line_length = 100
    tmp = []

    for idx, el in df.iterrows():
        lines = el['text'].split('\n')
        for line in lines:
            tmp.append({'text': line})
            if len(line) > line_length:
                line_length = len(line)
                print(f'Current longest line: {line_length}')
        if idx % 1000 == 0:
            length = len(df)
            print(f'Index at: {idx}/{length}')

    df = pd.DataFrame(tmp, columns=df_sentences.columns)

    print('Transforming text even more...')

    df['text'] = df.apply(lambda x: x.text.replace(u'\u00A0', ' '), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[\[\]]+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('\\\\', ' eller ', x['text']), axis=1)

    df['text'] = df.apply(lambda x: re.sub('[0-9]+[\S]+[A-Øa-ø]+', ' mix ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[A-Øa-ø]+[\S]+[0-9]+', ' mix ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(mix )+', ' mix ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('[0-9]+', ' number ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('(number )+', ' number ', x['text']), axis=1)

    df['text'] = df.apply(lambda x: re.sub('[^\s\dA-Øa-ø_]+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: re.sub('\s\s+', ' ', x['text']), axis=1)
    df['text'] = df.apply(lambda x: x['text'].strip().lower(), axis=1)

    print('Saving all the lines...')

    df = df[df['text'] != '']
    df.to_csv('request_tasktype_simple_word_embedding.csv', index=False)

    print(df)


def textInspect():
    table_request = 'request_tasktype_simple'
    df = pd.read_csv(f'{config.BASE_PATH}/cache/{table_request}_word_embedding.csv')
    df = df.fillna('')
    df = df[df['text'] != '']

    print(len(df))  # 2907952

    df['len'] = df.apply(lambda x: len(x['text']), axis=1)

    df = df[df['len'] <= 200]
    df = df[df['len'] >= 5]

    remove_n = 1000000
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df_subset = df.drop(drop_indices)

    df_subset.to_csv(f'{config.BASE_PATH}/cache/{table_request}_word_embedding_shortened.csv', index=False)

    print(len(df_subset))
    pass


def main():
    # Build text for Word Embedding
    # text()
    # textInspect()
    pass



if __name__ == '__main__':
    main()
