#!/usr/bin/env python3

import classifier
import file_utils
from numpy.random import seed
import parser
import sys
import tensorflow as tf

tf.random.set_random_seed(1)
seed(1)
module_name = 'es_humor.py'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'


def get_command_line_parameters(cmd_args):
    fun_name = 'get_command_line_parameters'
    error_msg_head = module_name + ' (' + fun_name + '): '
    if len(cmd_args) < 2:
        sys.exit(error_msg_head + ' Cantidad insuficiente de argumentos.')

    train_filename, val_filename, test_filename, embedding_filename = file_utils.check_data_path(cmd_args[1])

    test_file_list, out_file_list = file_utils.check_test_files(cmd_args[2:])

    return train_filename, val_filename, test_filename, embedding_filename, test_file_list, out_file_list


def main():
    cmd_args = sys.argv
    train_filename, val_filename, test_filename, embedding_filename,\
        test_file_list, out_file_list = get_command_line_parameters(cmd_args)

    # Create embedding matrix
    embedding_matrix, tokenizer = parser.embeddings_file2embeddings_matrix(embedding_filename)
    # Load training and validation data
    x_train_texts, y_train = parser.load_data(train_filename)
    x_val_texts, y_val = parser.load_data(val_filename)
    # Pre process training and validation data
    x_train = parser.preprocess_data(x_train_texts, tokenizer)
    x_val = parser.preprocess_data(x_val_texts, tokenizer)
    # Load classifier
    model = classifier.compile_classifier(embedding_matrix)
    # Train classifier on humor_train.csv
    classifier.train_model(model, x_train, y_train, x_val, y_val)
    # Load best model from training
    loaded_model = classifier.load_model()
    model = model if loaded_model is None else loaded_model
    # Evaluate classifier on humor_test.csv
    x_test_texts, y_test = parser.load_data(test_filename)
    x_test = parser.preprocess_data(x_test_texts, tokenizer)
    classifier.evaluate_model(model, x_test, y_test)
    # Test classifier on test_file1.csv,...,test_fileN.csv and save predictions on test_file1.out,...,test_fileN.out
    for i in range(len(test_file_list)):
        test_file = test_file_list[i]
        out_file = out_file_list[i]
        x_test_texts, y_test = parser.load_data(test_file)
        x_test = parser.preprocess_data(x_test_texts, tokenizer)
        y_out = classifier.evaluate_model(model, x_test, y_test)
        file_utils.save_array(y_out, out_file)
        print('Resulados guardados en {filename}'.format(filename=out_file))


if __name__ == '__main__':
    main()
