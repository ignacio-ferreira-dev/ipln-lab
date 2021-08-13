#!/usr/bin/env python3

import csv
import numpy as np
from numpy.random import seed
import parser
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
import time

tf.random.set_random_seed(1)
seed(1)
embedding_matrix = []


def create_model(dropout1=0.1, dropout2=0.1, recurrent_dropout1=0.1, recurrent_dropout2=0.1, activation1='relu',
                 activation2='relu', activation3='relu', optimizer='adam'):
    units = 50
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                        input_length=30, weights=[embedding_matrix], trainable=False, mask_zero=True))
    model.add(tf.keras.layers.GRU(units=2*units//5, dropout=dropout1, recurrent_dropout=recurrent_dropout1,
                                  kernel_initializer='glorot_uniform', activation=activation1, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=units//5, dropout=dropout2, recurrent_dropout=recurrent_dropout2,
                                  kernel_initializer='glorot_uniform', activation=activation2))
    model.add(tf.keras.layers.Dense(2*units//5, activation=activation3))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    WORD_EMBEDDINGS_FILENAME = 'data/intropln2019_embeddings_es_300.txt'
    DATA_TRAIN = 'data/humor_train.csv'
    DATA_VAL = 'data/humor_val.csv'

    # Create embedding matrix
    embedding_matrix, tokenizer = parser.embeddings_file2embeddings_matrix(WORD_EMBEDDINGS_FILENAME)
    # Load training and validation data
    x_train_texts, y_train = parser.load_data(DATA_TRAIN)
    x_val_texts, y_val = parser.load_data(DATA_VAL)
    # Preprocess training and validation data
    x_train = parser.preprocess_data(x_train_texts, tokenizer)
    x_val = parser.preprocess_data(x_val_texts, tokenizer)
    X = np.vstack((x_train, x_val))
    Y = np.append(y_train, y_val)

    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=25, batch_size=32, verbose=1)
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'Ftrl']
    activation1 = ['relu', 'elu', 'selu', 'tanh', 'softplus', 'softmax', 'softsign', 'sigmoid', 'hard_sigmoid']
    activation2 = ['relu', 'elu', 'selu', 'tanh', 'softplus', 'softmax', 'softsign', 'sigmoid', 'hard_sigmoid']
    activation3 = ['relu', 'elu', 'selu', 'tanh', 'softplus', 'softmax', 'softsign', 'sigmoid', 'hard_sigmoid']
    dropout1 = uniform(loc=0, scale=0.15)
    dropout2 = uniform(loc=0, scale=0.15)
    recurrent_dropout1 = uniform(loc=0, scale=0.15)
    recurrent_dropout2 = uniform(loc=0, scale=0.15)
    param_grid = dict(optimizer=optimizer, activation1=activation1, activation2=activation2, activation3=activation3,
                      dropout1=dropout1, dropout2=dropout2, recurrent_dropout1=recurrent_dropout1,
                      recurrent_dropout2=recurrent_dropout2)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, cv=3, n_iter=16, verbose=0)
    print('Start')
    start_time = time.time()
    grid_result = grid.fit(X, Y)
    end_time = time.time()
    print('Elapsed time (%1.1f min)' % ((end_time - start_time) / 60))
    # to check how data is being processed
    with open('output1.csv', 'w', newline='', encoding="utf-8") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([str(grid_result.best_params_)])
        wr.writerow([str(grid_result.best_score_)])

