import os
import re

import lemmy
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize

import config


class PreprocessingText:

    @staticmethod
    def get_str_from_tokens(tokens):
        return " ".join(str(x) for x in tokens)

    @staticmethod
    def get_tokens_from_str(string):
        return string.split(" ")

    @staticmethod
    def get_stopwords_removed(tokens, stopwords=None):
        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def get_lemma(lemmatizer, tokens):
        return [lemmatizer.lemmatize("", token)[0] for token in tokens]

    @staticmethod
    def get_tokenized_text(line, language="danish"):
        return [token.lower() for token in word_tokenize(line, language=language) if token.isalnum()]

    @staticmethod
    def get_beautiful_text(line):
        text = BeautifulSoup(line, "lxml").text
        text = re.sub('[\n.]', ' ', text)
        return text

    def __init__(self, df=None):
        print('[Status] Preprocessing text started')
        self.df = df
        self.stopwords = self.ready_stop_words()
        self.lemmatizer = lemmy.load("da")
        print('[Status] Preprocessing text ended')

    def ihlp(self):
        self.df['processText'] = self.df.apply(lambda x: self.get_process_text(' '.join(["{}".format(x[e]) for e in ['subject', 'description']])), axis=1)
        return self.df

    def get_process_text(
            self,
            text
    ):
        text = PreprocessingText.get_beautiful_text(text)
        tokens = PreprocessingText.get_tokenized_text(text)
        tokens = PreprocessingText.get_stopwords_removed(tokens=tokens, stopwords=self.stopwords)
        # tokens = PreprocessingText.get_lemma(tokens=tokens, lemmatizer=self.lemmatizer)
        string = PreprocessingText.get_str_from_tokens(tokens)
        return string

    def ready_stop_words(
            self,
            language='danish',
            file_path_input=f'{config.BASE_PATH}/logic/input/stopwords.txt',
    ):
        if os.path.isfile(file_path_input):
            stopwords = []
            with open(file_path_input, 'r') as file_handle:
                for line in file_handle:
                    currentPlace = line[:-1]
                    stopwords.append(currentPlace)
            return stopwords

        url = "http://snowball.tartarus.org/algorithms/%s/stop.txt" % language
        text = requests.get(url).text
        stopwords = re.findall('^(\w+)', text, flags=re.MULTILINE | re.UNICODE)

        with open(file_path_input, 'w') as file_handle:
            for list_item in stopwords:
                file_handle.write('%s\n' % list_item)

        return stopwords