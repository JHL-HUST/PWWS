# coding: utf-8
import os
import numpy as np
from config import config
import copy
import sys
from read_files import split_imdb_files, split_yahoo_files, split_agnews_files
import spacy
import argparse
import re
from collections import Counter, defaultdict

nlp = spacy.load('en')
parser = argparse.ArgumentParser('named entity recognition')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='yahoo')

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}


def recognize_named_entity(texts):
    '''
    Returns all NEs in the input texts and their corresponding types
    '''
    NE_freq_dict = copy.deepcopy(NE_type_dict)

    for text in texts:
        doc = nlp(text)
        for word in doc.ents:
            NE_freq_dict[word.label_][word.text] += 1
    return NE_freq_dict


def find_adv_NE(D_true, D_other):
    '''
    find NE_adv in D-D_y_true which is defined in the end of section 3.1
    '''
    # adv_NE_list = []
    for type in NE_type_dict.keys():
        # find the most frequent true and other NEs of the same type
        true_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_true[type]) if i < 15]
        other_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_other[type]) if i < 30]

        for other_NE in other_NE_list:
            if other_NE not in true_NE_list and len(other_NE.split()) == 1:
                # adv_NE_list.append((type, other_NE))
                print("'" + type + "': '" + other_NE + "',")
                with open('./{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
                    f.write("'" + type + "': '" + other_NE + "',\n")
                break


class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    imdb_0 = {'PERSON': 'David',
              'NORP': 'Australian',
              'FAC': 'Hound',
              'ORG': 'Ford',
              'GPE': 'India',
              'LOC': 'Atlantic',
              'PRODUCT': 'Highly',
              'EVENT': 'Depression',
              'WORK_OF_ART': 'Casablanca',
              'LAW': 'Constitution',
              'LANGUAGE': 'Portuguese',
              'DATE': '2001',
              'TIME': 'hours',
              'PERCENT': '98%',
              'MONEY': '4',
              'QUANTITY': '70mm',
              'ORDINAL': '5th',
              'CARDINAL': '7',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Lee',
              'NORP': 'Christian',
              'FAC': 'Shannon',
              'ORG': 'BAD',
              'GPE': 'Seagal',
              'LOC': 'Malta',
              'PRODUCT': 'Cat',
              'EVENT': 'Hugo',
              'WORK_OF_ART': 'Jaws',
              'LAW': 'RICO',
              'LANGUAGE': 'Sebastian',
              'DATE': 'Friday',
              'TIME': 'minutes',
              'PERCENT': '75%',
              'MONEY': '$',
              'QUANTITY': '9mm',
              'ORDINAL': 'sixth',
              'CARDINAL': 'zero',
              }
    imdb = [imdb_0, imdb_1]
    agnews_0 = {'PERSON': 'Williams',
                'NORP': 'European',
                'FAC': 'Olympic',
                'ORG': 'Microsoft',
                'GPE': 'Australia',
                'LOC': 'Earth',
                'PRODUCT': '#',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Chinese',
                'DATE': 'third-quarter',
                'TIME': 'Tonight',
                'MONEY': '#39;t',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '1',
                }
    agnews_1 = {'PERSON': 'Bush',
                'NORP': 'Iraqi',
                'FAC': 'Outlook',
                'ORG': 'Microsoft',
                'GPE': 'Iraq',
                'LOC': 'Asia',
                'PRODUCT': '#',
                'EVENT': 'Series',
                'WORK_OF_ART': 'Nobel',
                'LAW': 'Constitution',
                'LANGUAGE': 'French',
                'DATE': 'third-quarter',
                'TIME': 'hours',
                'MONEY': '39;Keefe',
                'ORDINAL': '2nd',
                'CARDINAL': 'Two',
                }
    agnews_2 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Baghdad',
                'LOC': 'Earth',
                'PRODUCT': 'Soyuz',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Constitution',
                'LANGUAGE': 'Filipino',
                'DATE': 'Sunday',
                'TIME': 'evening',
                'MONEY': '39;m',
                'QUANTITY': '20km',
                'ORDINAL': 'eighth',
                'CARDINAL': '6',
                }
    agnews_3 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Iraq',
                'LOC': 'Kashmir',
                'PRODUCT': 'Yukos',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'Gazprom',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Hebrew',
                'DATE': 'Saturday',
                'TIME': 'overnight',
                'MONEY': '39;m',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '6',
                }
    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]
    yahoo_0 = {'PERSON': 'Fantasy',
               'NORP': 'Russian',
               'FAC': 'Taxation',
               'ORG': 'Congress',
               'GPE': 'U.S.',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Hebrew',
               'DATE': '2004-05',
               'TIME': 'morning',
               'MONEY': '$ale',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'three',
               }
    yahoo_1 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_2 = {'PERSON': 'Equine',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'two',
               }
    yahoo_3 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_4 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Canada',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': 'hundred-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_5 = {'PERSON': 'Equine',
               'NORP': 'English',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Australia',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'MONEY': 'hundred-dollar',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',

               }
    yahoo_6 = {'PERSON': 'Fantasy',
               'NORP': 'Islamic',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_7 = {'PERSON': 'Fantasy',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'West',
               'PRODUCT': 'Variable',
               'EVENT': 'Watergate',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',
               }
    yahoo_8 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Chicago',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'QUANTITY': '$ale',
               'ORDINAL': 'Sixth',
               'CARDINAL': '2',

               }
    yahoo_9 = {'PERSON': 'Equine',
               'NORP': 'Chinese',
               'FAC': 'Music',
               'ORG': 'Digital',
               'GPE': 'U.S.',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '1918-1945',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '5'
               }
    yahoo = [yahoo_0, yahoo_1, yahoo_2, yahoo_3, yahoo_4, yahoo_5, yahoo_6, yahoo_7, yahoo_8, yahoo_9]
    L = {'imdb': imdb, 'agnews': agnews, 'yahoo': yahoo}


NE_list = NameEntityList()

if __name__ == '__main__':
    args = parser.parse_args()
    print('dataset:', args.dataset)
    class_num = config.num_classes[args.dataset]

    if args.dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        # get input texts in different classes
        pos_texts = train_texts[:12500]
        pos_texts.extend(test_texts[:12500])
        neg_texts = train_texts[12500:]
        neg_texts.extend(test_texts[12500:])
        texts = [neg_texts, pos_texts]
    elif args.dataset == 'agnews':
        texts = [[] for i in range(class_num)]
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        for i, label in enumerate(train_labels):
            texts[np.argmax(label)].append(train_texts[i])
        for i, label in enumerate(test_labels):
            texts[np.argmax(label)].append(test_texts[i])
    elif args.dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        texts = [[] for i in range(class_num)]
        for i, label in enumerate(train_labels):
            texts[np.argmax(label)].append(train_texts[i])
        for i, label in enumerate(test_labels):
            texts[np.argmax(label)].append(test_texts[i])

    D_true_list = []
    for i in range(class_num):
        D_true = recognize_named_entity(texts[i])  # D_true contains the NEs in input texts with the label y_true
        D_true_list.append(D_true)

    for i in range(class_num):
        D_true = copy.deepcopy(D_true_list[i])
        D_other = copy.deepcopy(NE_type_dict)
        for j in range(class_num):
            if i == j:
                continue
            for type in NE_type_dict.keys():
                # combine D_other[type] and D_true_list[j][type]
                for key in D_true_list[j][type].keys():
                    D_other[type][key] += D_true_list[j][type][key]
        for type in NE_type_dict.keys():
            D_other[type] = sorted(D_other[type].items(), key=lambda k_v: k_v[1], reverse=True)
            D_true[type] = sorted(D_true[type].items(), key=lambda k_v: k_v[1], reverse=True)
        print('\nfind adv_NE_list in class', i)
        with open('./{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
            f.write('\nfind adv_NE_list in class' + str(i))
        find_adv_NE(D_true, D_other)
