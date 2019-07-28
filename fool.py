# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import os
import numpy as np
from read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from word_level_process import word_process, get_tokenizer
from char_level_process import char_process
from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
from adversarial_tools import ForwardGradWrapper, adversarial_paraphrase
import tensorflow as tf
from keras import backend as K
import time
from unbuffered import Unbuffered

sys.stdout = Unbuffered(sys.stdout)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(
    description='Craft adversarial examples for a text classifier.')
parser.add_argument('--clean_samples_cap',
                    help='Amount of clean(test) samples to fool',
                    type=int, default=1000)
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')


def fool_text_classifier():
    clean_samples_cap = args.clean_samples_cap  # 1000
    print('clean_samples_cap:', clean_samples_cap)

    # get tokenizer
    dataset = args.dataset
    tokenizer = get_tokenizer(dataset)

    # Read data set
    x_test = y_test = None
    test_texts = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)

    # Write clean examples into a txt file
    clean_texts_path = r'./fool_result/{}/clean_{}.txt'.format(dataset, str(clean_samples_cap))
    if not os.path.isfile(clean_texts_path):
        write_origin_input_texts(clean_texts_path, test_texts)

    # Select the model and load the trained weights
    assert args.model[:4] == args.level
    model = None
    if args.model == "word_cnn":
        model = word_cnn(dataset)
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
    elif args.model == "word_lstm":
        model = lstm(dataset)
    model_path = r'./runs/{}/{}.dat'.format(dataset, args.model)
    model.load_weights(model_path)
    print('model path:', model_path)

    # evaluate classification accuracy of model on clean samples
    scores_origin = model.evaluate(x_test[:clean_samples_cap], y_test[:clean_samples_cap])
    print('clean samples origin test_loss: %f, accuracy: %f' % (scores_origin[0], scores_origin[1]))
    all_scores_origin = model.evaluate(x_test, y_test)
    print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))

    grad_guide = ForwardGradWrapper(model)
    classes_prediction = grad_guide.predict_classes(x_test[: clean_samples_cap])

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    sub_rate_list = []
    NE_rate_list = []

    start_cpu = time.clock()
    adv_text_path = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, str(clean_samples_cap))
    change_tuple_path = r'./fool_result/{}/{}/change_tuple_{}.txt'.format(dataset, args.model, str(clean_samples_cap))
    file_1 = open(adv_text_path, "a")
    file_2 = open(change_tuple_path, "a")
    for index, text in enumerate(test_texts[: clean_samples_cap]):
        sub_rate = 0
        NE_rate = 0
        if np.argmax(y_test[index]) == classes_prediction[index]:
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(input_text=text,
                                                                                          true_y=np.argmax(y_test[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level=args.level)
            if adv_y != np.argmax(y_test[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)
            file_2.write(str(index) + str(change_tuple_list) + '\n')
        file_1.write(text + " sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")
    end_cpu = time.clock()
    print('CPU second:', end_cpu - start_cpu)
    mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
    mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
    print('mean substitution rate:', mean_sub_rate)
    print('mean NE rate:', mean_NE_rate)
    file_1.close()
    file_2.close()


if __name__ == '__main__':
    args = parser.parse_args()
    fool_text_classifier()
