# coding: utf-8
from config import config
import copy
import spacy
from word_level_process import text_to_vector
from char_level_process import doc_process, get_embedding_dict

nlp = spacy.load('en_core_web_sm')


def evaluate_word_saliency(doc, grad_guide, tokenizer, input_y, dataset, level):
    word_saliency_list = []

    # zero the code of the current word and calculate the amount of change in the classification probability
    if level == 'word':
        max_len = config.word_max_len[dataset]
        text = [doc[position].text for position in range(len(doc))]
        text = ' '.join(text)
        origin_vector = text_to_vector(text, tokenizer, dataset)
        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        for position in range(len(doc)):
            if position >= max_len:
                break
            # get x_i^(\hat)
            without_word_vector = copy.deepcopy(origin_vector)
            without_word_vector[0][position] = 0
            prob_without_word = grad_guide.predict_prob(input_vector=without_word_vector)

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, doc[position], word_saliency, doc[position].tag_))

    elif level == 'char':
        max_len = config.char_max_len[dataset]
        embedding_dic = get_embedding_dict()
        origin_vector = doc_process(doc.text.lower(), embedding_dic, dataset).reshape(1, max_len)
        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)

        find_a_word = False
        word_position = 0
        without_word_vector = copy.deepcopy(origin_vector)
        for i, c in enumerate(doc.text):
            if i >= max_len:
                break
            if c is not ' ':
                without_word_vector[0][i] = 0
            else:
                find_a_word = True
                prob_without_word = grad_guide.predict_prob(without_word_vector)
                word_saliency = origin_prob[input_y] - prob_without_word[input_y]
                word_saliency_list.append((word_position, doc[word_position], word_saliency, doc[word_position].tag_))
                word_position += 1
            if find_a_word:
                without_word_vector = copy.deepcopy(origin_vector)
                find_a_word = False

    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list
