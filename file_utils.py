#!/usr/bin/env python3

import numpy as np
import os.path
import sys

module_name = 'file_utils.py'


def check_data_path(data_path):
    fun_name = 'check_data_path'
    error_msg_head = module_name + ' (' + fun_name + '): '

    if data_path[-1] != '/':
        data_path += '/'
    if os.path.isdir(data_path):
        train_filename = data_path + 'humor_train.csv'
        val_filename = data_path + 'humor_val.csv'
        test_filename = data_path + 'humor_test.csv'
        embedding_filename = data_path + 'intropln2019_embeddings_es_300.txt'
        # Check train, val and test files
        if not os.path.isfile(train_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de entrenamiento ' + train_filename)
        if not os.path.isfile(val_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de validación ' + val_filename)
        if not os.path.isfile(test_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de testing ' + test_filename)
        if not os.path.isfile(embedding_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de testing ' + embedding_filename)
    return train_filename, val_filename, test_filename, embedding_filename


def check_test_files(test_files_list):
    fun_name = 'check_test_files'
    error_msg_head = module_name + ' (' + fun_name + '): '

    test_file_list = []
    out_file_list = []
    for test_file in test_files_list:
        if not os.path.isfile(test_file):
            sys.exit(error_msg_head + ' No se encontró el archivo de test ' + test_file)
        filename, extension = os.path.splitext(test_file)
        if extension != '.csv':
            sys.exit(error_msg_head + ' El archivo de test no tiene extensión .csv' + test_file)
        test_file_list.append(test_file)
        out_file_list.append(filename + '.out')
    return test_file_list, out_file_list


def save_array(arr, filename):
    np.savetxt(filename, arr, fmt='%1d', delimiter='\n', encoding='utf-8')
