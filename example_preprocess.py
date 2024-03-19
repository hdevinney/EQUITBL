#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 26.04.2022
# author: Hannah Devinney
# preprocess the brown corpus to bags of lemmaPOSes (chunks into documents using sliding window!!); save to gensim corpus/dict
# note that some preprocessing info MUST be in the config file!


import io
import pickle
import re
import pandas as pd
from configparser import ConfigParser
from argparse import ArgumentParser

#preprocessing (for English) is in equitbl/tools/corpus/preprocessing_eng
from tools.corpus import preprocessing_eng as preproc

from gensim import corpora
import gensim

def main():
    ##################### ARGUMENTS #######################
    a = ArgumentParser()
    a.add_argument('-config', dest='config_file', required=True, type=str, help="path to .ini configuration file (for specifying file paths)")
    opts = a.parse_args()
    config = ConfigParser()
    config.read(opts.config_file)

    #paths
    base_dir = config['FILE_PATHS']['project_root']
    input_dir = base_dir + 'input/corpora/'
    output_dir = base_dir + 'models/bow/'

    #files
    seedfile = base_dir + 'input/seeds/' + config['FILE_PATHS']['seed_file']
    input_corpus = input_dir + config['FILE_PATHS']['corpus_name'] + '.json'
    output_name = config['FILE_PATHS']['dictionary_name']

    #preprocessing choices
    CHUNK_SIZE = config['PREPROCESSING'].getint('chunk_size')
    MINIMUM = config['PREPROCESSING'].getint('minimum_freq')
    IGNORE = config['PREPROCESSING']['ignore_pos'].replace(" ", "").strip('[]').split(',')
    STOPWORDS = config['PREPROCESSING']['stopwords'].replace(" ", "").strip('[]').split(',')

    ########################################################

    #get seed words (will be exempted from pruning)
    with open(seedfile, 'r') as inputfile:
        seed_list = inputfile.readlines()
        seed_list = [seed.rstrip() for seed in seed_list] #clean up whitespace

    #process the input documents (get TOKENS (tagged and lemmatized))
    print("processing: {}".format(input_corpus))
    df = pd.read_json(open(input_corpus))
    lemma_dictionary = preproc.get_chunked_pos_lemmas_dictionary(df, IGNORE, STOPWORDS, CHUNK_SIZE)

    #prune vocabulary
    print("PRUNING INFREQUENT NON-SEED TERMS")
    init_docs, frequency = preproc.get_docs_and_frequencies(lemma_dictionary)
    documents = preproc.prune_dict(init_docs, frequency, seed_list, MINIMUM)
    print("completed")

    #convert to gensim dictionary; save
    dictionary = corpora.Dictionary(documents)
    dictionary.save(output_dir + output_name + ".gensim")
    print("Constructed dictionary")

    #process the documents; save info as a gensim corpus
    corpus = [dictionary.doc2bow(text) for text in documents]
    print("Constructed corpus BoW: " + str(len(corpus)))
    pickle.dump(corpus, open(output_dir + output_name + ".pkl", 'wb'), protocol=2)

    print("Saved dictionary and corpus in " + output_dir)

    #you may also want a .mm format for the corpus (for back up etc.)
    #if so, uncomment the following:
#    corpora.MmCorpus.serialize(output_dir + output_name + ".mm", corpus)
#    print("saved gensim version of corpus to " + output_dir + output_name + ".mm")



    
if __name__ == "__main__":
    main()
