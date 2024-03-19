#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 1.0 (2.09.2021)
# author: Hannah Devinney
# a nice(?) way to look at term frequencies in a particular corpus
#

import os
import csv

import gensim
from gensim import corpora
import pickle
import numpy


def load_dictionary(filepath):
    return gensim.corpora.Dictionary.load(filepath)

def load_corpus(filepath):
    return pickle.load(open(filepath, 'rb'))


def get_freq(word, dictionary):
    term = dictionary.token2id[word]
    count_freq = dictionary.cfs[term]
    doc_freq = dictionary.dfs[term]
    return count_freq, doc_freq


#interactive mode (specify a term, get a frequency)
def frequencies_interactive(dictionary):
    term = ""
    print("REMEMBER: terms may be lemmatized and/or include POS tags!")
    while term != "q":
        #ask user for a term or 'q'
        term = input("please specify a term or type 'q' to quit:\t")
        freq, docs = get_freq(term, dictionary)
        print("The term {0} appears {1} times, over {2} unique documents.\n".format(term, freq, docs))
    print("---End of Interactive Mode---")
        
  
#read from seed word list
def get_frequencies(word_list, dictionary):
    freq_dict = {}
    for word in wordlist:
        count, doc = get_freq(word, dictionary)
        freq_dict[word] = (count, doc)
    return freq_dict

#just get everything in dictionary and save in a csv file or whatever
def get_all_frequencies(dictionary, outfile):
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writewrow(["term", "count_freq", "doc_freq"])
        for term in dictionary:
            word = dictionary[term]
            count_freq = dictionary.cfs[term]
            doc_freq = dictionary.dfs[term]
            writer.writerow([word, count_freq, doc_freq])
        
