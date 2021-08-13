#!/usr/bin/env python3

from constants import *
from hyperparameters import *
import numpy as np
import pandas as pd
import re
import statistics
import stopwords
import tensorflow as tf


def load_data(data_filename):
    print('Leyendo archivo {filename}...'.format(filename=data_filename))
    df = pd.read_csv(data_filename)
    if 'humor' in df.columns:
        return np.array(df['text']), np.array(df['humor'])
    else:
        return np.array(df['text']), None


def _space_non_alphanumeric(text):
    r = re.compile('([^a-zA-Z0-9ñÍÓÚÉÁ \t\n\r\f\váéíóú])')
    return r.sub(r' \1 ', text)


def _remove_stop_words(text):
    text_words = text.split(' ')
    text_words = list(filter(lambda w: w not in stopwords.stopwords_dict, text_words))
    return ' '.join(text_words)


def _get_word_embeddings(embeddings_filename):
    word_embedding = {}
    # save indexes 0 and 1 for padding and unknown words
    with open(embeddings_filename, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # little hack to catch 4 four words (special characters)
            if len(coefs) != 300:
                word = line[0]
                coefs = line[1:]
                coefs = np.fromstring(coefs, 'f', sep=' ')
            word_embedding[word] = coefs
    return word_embedding


def _get_embedding_vector(word_embedding, word):
    if word in word_embedding:
        return word_embedding[word]
    elif word.capitalize() in word_embedding:
        return word_embedding[word.capitalize()]
    elif word.lower() in word_embedding:
        return word_embedding[word.lower()]
    elif word.upper() in word_embedding:
        return word_embedding[word.upper()]
    else:
        return None


def embeddings_file2embeddings_matrix(embeddings_filename):
    print('Computando matriz de embeddings...')
    word_embedding = _get_word_embeddings(embeddings_filename)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=EMBEDDINGS_NUM_WORDS, filters='',
                                                      lower=False, split=' ', char_level=False, oov_token='<UNK>')
    tokenizer.fit_on_texts(list(word_embedding.keys()))
    embeddings_word_index = tokenizer.word_index
    # save index 0 and 1 for padding and unknown words
    embedding_matrix = np.zeros((len(word_embedding) + 2, EMBEDDING_VECTOR_SIZE))
    for word, index in embeddings_word_index.items():
        if index != 1:
            embedding_vector = _get_embedding_vector(word_embedding, word)
            embedding_matrix[index] = embedding_vector
    return embedding_matrix, tokenizer


def preprocess_data(texts_list, tokenizer):
    print('Pre procesando textos...')
    texts_list = list(map(lambda x: _space_non_alphanumeric(x), texts_list))
    texts_list = list(map(lambda x: _remove_stop_words(x), texts_list))
    indexes_list = tokenizer.texts_to_sequences(texts_list)
    indexes_list = tf.keras.preprocessing.sequence.pad_sequences(indexes_list, maxlen=MAX_WORDS_PER_TWEET)
    return np.array(indexes_list)


if __name__ == '__main__':
    texts_list, _ = load_data('data/humor_train.csv')
    texts_list = list(map(lambda x: _space_non_alphanumeric(x), texts_list))
    texts_list = list(map(lambda x: x.split(' '), texts_list))
    num_words_list = np.array(list(map(lambda x: len(x), texts_list)))
    median_num_words = statistics.median(num_words_list)
    print('Median word number: ', median_num_words)
    deviation = statistics.median(np.abs(num_words_list - statistics.median(num_words_list)))
    print('Deviation: ', deviation)
