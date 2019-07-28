class Config(object):
    num_classes = {'imdb': 2, 'yahoo': 10, 'agnews': 4}
    word_max_len = {'imdb': 400, 'yahoo': 1000, 'agnews': 150}
    char_max_len = {'agnews': 1014}
    num_words = {'imdb': 5000, 'yahoo': 20000, 'agnews': 5000}

    wordCNN_batch_size = {'imdb': 32, 'yahoo': 32, 'agnews': 32}
    wordCNN_epochs = {'imdb': 2, 'yahoo': 6, 'agnews': 2}

    bdLSTM_batch_size = {'imdb': 32, 'yahoo': 32, 'agnews': 64}
    bdLSTM_epochs = {'imdb': 6, 'yahoo': 16, 'agnews': 2}

    charCNN_batch_size = {'agnews': 128}
    charCNN_epochs = {'agnews': 4}

    LSTM_batch_size = {'imdb': 32, 'agnews': 64}
    LSTM_epochs = {'imdb': 30, 'agnews': 30}

    loss = {'imdb': 'binary_crossentropy', 'yahoo': 'categorical_crossentropy', 'agnews': 'categorical_crossentropy'}
    activation = {'imdb': 'sigmoid', 'yahoo': 'softmax', 'agnews': 'softmax'}

    wordCNN_embedding_dims = {'imdb': 50, 'yahoo': 50, 'agnews': 50}
    bdLSTM_embedding_dims = {'imdb': 128, 'yahoo': 128, 'agnews': 128}
    LSTM_embedding_dims = {'imdb': 100, 'agnews': 100}


config = Config()
