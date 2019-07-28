# coding=utf-8
import numpy as np
from config import config


def onehot_dic_build():
    # use one-hot encoding
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    embedding_dic = {}
    embedding_w = []
    # For characters that do not exist in the alphabet or empty characters, replace them with vectors 0.
    embedding_dic["UNK"] = 0
    embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

    for i, alpha in enumerate(alphabet):
        onehot = np.zeros(len(alphabet), dtype='float32')
        embedding_dic[alpha] = i + 1
        onehot[i] = 1
        embedding_w.append(onehot)

    embedding_w = np.array(embedding_w, dtype='float32')
    return embedding_w, embedding_dic


def get_embedding_dict():
    return {'UNK': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
            'k': 11, 'l': 12,
            'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
            'w': 23, 'x': 24,
            'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34,
            '8': 35, '9': 36,
            '-': 60, ',': 38, ';': 39, '.': 40, '!': 41, '?': 42, ':': 43, "'": 44, '"': 45, '/': 46,
            '\\': 47, '|': 48,
            '_': 49, '@': 50, '#': 51, '$': 52, '%': 53, '^': 54, '&': 55, '*': 56, '~': 57, '`': 58,
            '+': 59, '=': 61,
            '<': 62, '>': 63, '(': 64, ')': 65, '[': 66, ']': 67, '{': 68, '}': 69}


def doc_process(doc, embedding_dic, dataset):
    max_len = config.char_max_len[dataset]
    min_len = min(max_len, len(doc))
    doc_vec = np.zeros(max_len, dtype='int64')
    for j in range(min_len):
        if doc[j] in embedding_dic:
            doc_vec[j] = embedding_dic[doc[j]]
        else:
            doc_vec[j] = embedding_dic['UNK']
    return doc_vec


def doc_process_for_all(doc, embedding_dic, dataset):
    max_len = config.char_max_len[dataset]
    x = []
    for d in doc:
        x.append(doc_process(d, embedding_dic, dataset))
    x = np.asarray(x, dtype='int64')
    return x


def char_process(train_texts, train_labels, test_texts, test_labels, dataset):
    embedding_w, embedding_dic = onehot_dic_build()

    x_train = []
    for i in range(len(train_texts)):
        doc_vec = doc_process(train_texts[i].lower(), embedding_dic, dataset)
        x_train.append(doc_vec)
    x_train = np.asarray(x_train, dtype='int64')
    y_train = np.array(train_labels, dtype='float32')

    x_test = []
    for i in range(len(test_texts)):
        doc_vec = doc_process(test_texts[i].lower(), embedding_dic, dataset)
        x_test.append(doc_vec)
    x_test = np.asarray(x_test, dtype='int64')
    y_test = np.array(test_labels, dtype='float32')

    del embedding_w, embedding_dic
    return x_train, y_train, x_test, y_test
