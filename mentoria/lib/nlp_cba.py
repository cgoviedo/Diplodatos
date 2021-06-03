import glob
import os
import pandas as pd
from collections import Counter
from nltk.util import ngrams

# Processor que reemplaza varios caracteres contra uno solo

class MapMultipleCharsToCharProcessor:

    def __init__(self, characters_to_replace, new_value = " " , to_lower = False):
        self.characters_to_replace = characters_to_replace
        self.new_value = new_value
        self.to_lower = to_lower

    def process(self, text):
        for ch in self.characters_to_replace:
            text = text.replace(ch,self.new_value)

        if self.to_lower:
            text = text.lower()
        return text


# Implementa un visitor. Se registran N procesors los cuales se ejecutan en el orden que fueron
# agregados.

class TextCleaner:

    def __init__(self):
        self.processors = []

    def registerProcessor(self, processor):
        self.processors.append(processor)


    def process(self, text):

        for processor in self.processors:
                text = processor.process(text)

        return text


# Tokeniza el string usando "tokenizer_symbol"

def tokenize (text, tokenizer_symbol):
    result = text.split(tokenizer_symbol)

    return result


def generate_corpus_df(directories, text_cleaner):
    corpus = []

    for directory in directories:

        file_list = glob.glob(os.path.join(os.getcwd(), directory, "*.txt"))


        for file_path in file_list:
            with open(file_path) as f_input:
                corpus.append([text_cleaner.process(f_input.read()) , directory])

    return pd.DataFrame(corpus, columns=["text", "classifier"] )


def get_words(text):
    words = tokenize(text, " ")
    words = [ word for word in words if not word.strip() == ""]
    #words = [ word for word in words ]
    return words


def get_words_ocrruence(words):

    word_freq = Counter(words)
    return pd.DataFrame(word_freq.most_common() , columns= ["word" , "count"])


def get_word_ocurrence_df(aggregated_corups_df, classifier_name):

    text = aggregated_corups_df[aggregated_corups_df['classifier'] == classifier_name ]['text'].values[0]

    words = get_words(text)
    df = get_words_ocrruence(words)
    return (df, words)


def get_anagrams(words, n):

    result = ngrams(words, n)
    words = Counter(result)
    return words

def get_anagrams_count_df(words, n_gram_count):

    words = get_anagrams(words,n_gram_count )
    df = get_words_ocrruence(words)
    return df

def get_df_word_cloud(df, column, column_order, limit_words = 20):

    sorted_df = df.sort_values(column_order ,ascending=False)[0:limit_words]
    sorted_df["word_"] = sorted_df.apply( lambda row:" ".join(row[column] ), axis=1)

    return sorted_df
