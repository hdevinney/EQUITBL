#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 1.5ish? (9.9.2021)
# author: Hannah Devinney
# runs pSSLDA; saves topic models (hopefully) 
#gave up on relative paths, this seems to work, but will need fixing...


import pdb

import os
import sys

from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as NP
import numpy.random as NPR

from pSSLDA import infer #MAKE SURE these are the py3 versions
import FastLDA #ESPECIALLY THIS ONE!!

import gensim
from gensim import corpora
import pickle

#gave up on relative import, hence the relative path situation here
sys.path.append('..')
import model_printing as mp


def load_dictionary():
    return gensim.corpora.Dictionary.load(BASE_DIRECTORY + 'models/bow/' + DICTIONARY_NAME + '.gensim')

def load_corpus():
    print(BASE_DIRECTORY + 'models/bow/' + CORPUS_NAME + '.pkl')
    return pickle.load(open(BASE_DIRECTORY + 'models/bow/' + CORPUS_NAME + '.pkl', 'rb'))

def get_indices(file_name, dict):
    list_file = open(WORD_LIST_DIRECTORY + file_name)
    word_list = []
    for line in list_file:
        line = line.replace('\n', '')
        word_list.append(line)
    list_file.close()
    indices = dict.doc2idx(word_list, unknown_word_index=-1)
    indices = [i for i in indices if i >= 0]
    return indices

def main():
    ##################### READ IN FOR BATCH #######################
    a = ArgumentParser()
    a.add_argument('-config', dest='config_file', required=True, type=str, help="path to .ini configuration file")

    opts = a.parse_args()
    config = ConfigParser()
    config.read(opts.config_file)

    #set global arguments
    #I suspect this isn't a particularly pythonic solution... 
    global CORPUS_NAME
    CORPUS_NAME = config['FILE_PATHS']['corpus_name']
    global DICTIONARY_NAME
    DICTIONARY_NAME = config['FILE_PATHS']['dictionary_name']
    global BASE_DIRECTORY
    BASE_DIRECTORY = config['FILE_PATHS']['project_root']
    global WORD_LIST_DIRECTORY
    WORD_LIST_DIRECTORY = BASE_DIRECTORY + "input/seeds/"
    global OUTFILE_NAME
    output_name = config['FILE_PATHS']['output_name']
    OUTFILE_NAME = BASE_DIRECTORY + "output/raw/" + output_name
    MODEL_NAME = BASE_DIRECTORY + "models/tm/" + output_name

    ################################################################

    
    print('Loading dictionary')
    dict = load_dictionary()
    print('Loading corpus')
    corpus = load_corpus()
       
    T = config['HYPER_PARAMS'].getint('t')  # number of topics
    W = len(dict.keys())             # vocabulary size
    N = dict.num_pos                 # corpus length
    D = dict.num_docs                # number of documents

    print("****\n{} Topics\n{} Terms in vocab\n{} Tokens in Corpus\n{} Documents in Corpus\n****".format(T,W,N,D))

    (wordvec, docvec) = ([],[])
    d = 0
    # construct word and document vectors from corpus
    print('Constructing word and doc vectors')
    for doc in corpus:
        for (word, count) in doc:
            for i in range(count):
                wordvec.append(word)
                docvec.append(d)
        d += 1
    # convert wordvec and docvec to numpy arrays
    (w, d) = (NP.array(wordvec, dtype = NP.int), NP.array(docvec, dtype = NP.int))

     
    # Create parameters
    alpha_weight = config['HYPER_PARAMS'].getfloat('alpha_weight')
    beta_weight = config['HYPER_PARAMS'].getfloat('beta_weight')
    alpha = NP.ones((1,T)) * alpha_weight
    beta = NP.ones((T,W)) * beta_weight

    P = config['HYPER_PARAMS'].getint('samplers') # number of parallel samplers
    print("Inference array will be of size {} x {}".format(N,P))

    randseed = config['HYPER_PARAMS'].getint('random_seed')  # random seed
    numsamp = config['HYPER_PARAMS'].getint('samples') # number of samples to take

    # Add z labels
    label_weight = config['HYPER_PARAMS'].getfloat('z')

    #assign seeded topics
    #tells us how many topics to seed
    wlists = config['FILE_PATHS'].get('word_lists').split(',')
    num_seeded = len(wlists)
    #each 'label' in labels is an array of T 0s
    labels = []
    for i in range(0, num_seeded):
        #initialize 'label'
        l = NP.zeros((T,), dtype=float)
        #for every seeded topic, initialize w/ z-weight
        print("setting position {} in {} to {}".format(i, l, label_weight))
        l[i] = label_weight
        labels.append(l)
    print(labels)

    #in comparison:
    lw = 5.0
    label0 = NP.zeros((T,), dtype=float)
    label0[0] = lw
    label1 = NP.zeros((T,), dtype=float)
    label1[1] = lw
    label2 = NP.zeros((T,), dtype=float)
    label2[2] = lw

#    print("label 0 should be: {}".format(label0))
#    print("label 1 should be: {}".format(label1))
#    print("label 2 should be: {}".format(label2))

    #wordlists is a list of [lists of indices]
    #(ids of each seed term per doc2idx)
    wordlists = []
    tmp = 0
    for wl in wlists:
        #TODO: comment this out / find a place to store it secretly (to check results without "biasing" them with expectations)
        print("topic {} corresponds to : {}".format(tmp, wl))
        tmp += 1
        indices = get_indices(wl, dict)
        wordlists.append(indices)
    print("set word lists: {}".format(wordlists)) #just to check
    print("terms: {}".format([[dict.get(idx) for idx in wordlist] for wordlist in wordlists]))
    print("setting up zlabels")

    #TODO: add a check ensuring that word lists do not overlap
    

    #set up zlabels
    #TODO: add a way to include the same seed word "split" across multiple lists (lower priority than non-overlapping)
    zlabels = [] 
    counts = [0] * num_seeded #(previously fcount, mcount, ncount, etc.)
    print("term \tdict_id \tzlabel")
    for wi in w:
        in_a_list = False #reset every time just in case
        for i in range(0, num_seeded):  #check each list of seed idxs
            if(wi in wordlists[i]): 
                zlabels.append(labels[i])
                counts[i] += 1
                in_a_list = True #ensure None doesn't get added this time
                break #don't check any more seed lists (since we for now assume a term can belong to only one seed list)
        if not in_a_list:
            zlabels.append(None) #zlabels should align to w (word vector)
           
    print("**********************************finished zlabels*******************************")
    print(len(zlabels))

    #set up zcomparisons (FOR TESTING TO MAKE SURE NOTHING IS GOING WRONG WITH MAKING THE VECTORS)
    '''zcomps = []
    indices0 = get_indices("gendered_lemmaPOS/fem_seed.txt", dict)
    indices1 = get_indices("gendered_lemmaPOS/masc_seed.txt", dict)
    indices2 = get_indices("gendered_lemmaPOS/neu_seed.txt", dict)
    print(indices0)
    print(indices1)
    print(indices2)
    for wi in w:
        if (wi in indices0):
            zcomps.append(label0)
        elif(wi in indices1):
            zcomps.append(label1)
        elif(wi in indices2):
            zcomps.append(label2)
        else:
            zcomps.append(None)'''
            
    tmp = 0
    for count in counts:
        print("list {} count: {}".format(tmp, count))
        tmp += 1

    '''print("********************** LAST 1000 ZLABELS *************************")
    i = -1000
    while i < 0:
        print("word_idx: {} term: {} doc: {} zlabel: {}".format(w[i], dict.get(w[i]), d[i], zlabels[i]))
        i += 1

    print("********************** LAST 1000 ZL-COMPS *************************")
    i = -1000
    while i < 0:
        print("word_idx: {} term: {} doc: {} zlabel: {}".format(w[i], dict.get(w[i]), d[i], zcomps[i]))
        i += 1'''

####    sys.exit() #quit early bc we're just testing zlabels
    
    # Do parallel inference
    print('Doing parallel inference')
    finalz = infer(w, d, alpha, beta, numsamp, randseed, P, zlabels = zlabels)

    #TODO ADD THIS OPTION BACK IN I GUESS
    #else:
        # Do UNSUPERVIZED parallel inference
#    print('Doing unsupervized parallel inference')
#    finalz = infer(w, d, alpha, beta, numsamp, randseed, P)#, zlabels = zlabels)
        #(remove zlabels=zlabels to do unweighted)

    # Estimate phi and theta
    print( 'Estimating phi and theta')
    (nw, nd) = FastLDA.countMatrices(w, W, d, D, finalz, T)
    (estphi,esttheta) = FastLDA.estPhiTheta(nw, nd, alpha, beta)

    print('')
    print('Number of documents')
    print(dict.num_docs)
    print('Corpus length')
    print(dict.num_pos)
    print('Vocabulary size')
    print(len(dict.keys()))
    print('Number of topics')
    print(str(T))
    print('Length of phi')
    print(len(estphi))
    print('Length of theta')
    print(len(esttheta))
    print('')
    
    #print(estphi)

    save_topics = OUTFILE_NAME + ".txt"
    mp.print_topics(estphi, dict, 10)
    mp.print_to_file(estphi, dict, 50, save_topics)

    if config['FILE_PATHS']['log_file'] is not None:
        mp.log_counts(config['FILE_PATHS']['log_file'], counts, len(estphi), len(esttheta))


    #SAVE MODELS
    save_phi = MODEL_NAME + "_phi"
    save_theta = MODEL_NAME + "_theta"
    print('\n\nSaving topic models to {}'.format(OUTFILE_NAME))
    NP.save(save_phi, estphi)
    NP.save(save_theta, esttheta)
    print("done!")

        
    
if __name__ == "__main__":
    main()

