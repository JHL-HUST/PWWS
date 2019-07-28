# coding: utf-8
from __future__ import print_function
from config import config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D, Flatten
from word_level_process import get_tokenizer
import numpy as np


def get_embedding_index(file_path):
    global embeddings_index
    embeddings_index = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))


def get_embedding_matrix(dataset, num_words, embedding_dims):
    # global num_words, embedding_matrix, word_index
    global embedding_matrix, word_index
    word_index = get_tokenizer(dataset).word_index
    print('Preparing embedding matrix.')
    # num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dims))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def word_cnn(dataset, use_glove=False):
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()

    if use_glove:
        file_path = r'./glove.6B.{}d.txt'.format(str(embedding_dims))
        get_embedding_index(file_path)
        get_embedding_matrix(dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # for CNN_2
    # model.add(Dense(hidden_dims))
    # # model.add(Dropout(0.2))
    # model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def bd_lstm(dataset):
    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.bdLSTM_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_bdlstm model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Bidirectional(LSTM(64)))  # 64 / LSTM-2:128 / LSTM-3: 32
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation))

    # try using different optimizers and different optimizer configs
    model.compile('adam', loss, metrics=['accuracy'])
    return model


def lstm(dataset, use_glove=True):
    drop_out = 0.3

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.LSTM_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_lstm model...')
    model = Sequential()
    if use_glove:
        file_path = r'./glove.6B.' + str(embedding_dims) + 'd.txt'
        get_embedding_index(file_path)
        get_embedding_matrix(dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))

    model.add(LSTM(128, name="lstm_layer", dropout=drop_out, recurrent_dropout=drop_out))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation, name="dense_one"))

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model


def char_cnn(dataset):
    max_len = config.char_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]

    print('Build char_cnn model...')
    model = Sequential()

    model.add(Embedding(70, 69, input_length=max_len))

    model.add(Conv1D(256, 7, padding='valid', activation='relu', strides=1))
    model.add(MaxPool1D(3))

    model.add(Conv1D(256, 7, padding='valid', activation='relu', strides=1))
    model.add(MaxPool1D(3))

    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPool1D(3))

    model.add(Flatten())

    model.add(Dense(max_len))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(max_len))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
