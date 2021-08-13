#!/usr/bin/env python3

from metrics import Metrics
import numpy as np
from hyperparameters import *
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import time


def compile_classifier(embedding_matrix):
    print('Compilando clasificador...')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                        input_length=MAX_WORDS_PER_TWEET, weights=[embedding_matrix], trainable=False,
                                        mask_zero=True))
    model.add(tf.keras.layers.GRU(units=2*TOTAL_UNITS//5, dropout=DROPOUT1, recurrent_dropout=RECURRENT_DROPOUT1,
                                  kernel_initializer=KERNEL_INITIALIZER1, activation=ACTIVATION1, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=TOTAL_UNITS//5, dropout=DROPOUT2, recurrent_dropout=RECURRENT_DROPOUT2,
                                  kernel_initializer=KERNEL_INITIALIZER2, activation=ACTIVATION2))
    model.add(tf.keras.layers.Dense(2*TOTAL_UNITS//5, activation=ACTIVATION3))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    print('Entrenando modelo...')
    metrics = Metrics(x_val, y_val)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                                    mode='max', period=1)
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, workers=2, use_multiprocessing=True,
              validation_freq=1, validation_data=(x_val, y_val), callbacks=[metrics, checkpoint], verbose=1)
    end_time = time.time()
    print('Entrenamiento finalizado (%1.1f minutos)' % ((end_time - start_time) / 60))


def load_model():
    if os.path.isfile('model.h5'):
        print('Cargando el mejor modelo obtenido durante el entrenamiento...')
        return tf.keras.models.load_model('model.h5')
    else:
        print('WARNING: no se encontró el archivo model.h5 del mejor modelo del entrenamiento.')
        return None


def evaluate_model(model, x_test, y_test=None):
    print('Evaluando modelo...')
    y_out = np.asarray(model.predict_classes(x_test, batch_size=BATCH_SIZE, verbose=1))
    if y_test is not None:
        print('Precision: ', precision_score(y_test, y_out))
        print('Recall: ', recall_score(y_test, y_out))
        print('F1 Score: ', f1_score(y_test, y_out))
    return y_out
