#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 14.03.2022
# author: Hannah Devinney (with some code taken and/or adapted from Henrik Björklund)
# toolkit for evaluating individual documents (likelihood each topic in a TM is to have produced it; splitting these documents into files in the first place)

import pandas as pd
import numpy as np
import math
from gensim import corpora
import gensim
import os
from os import listdir
from os.path import isfile, join
#preprocessing is in equitbl/tools/corpus/preprocessing_eng 
from tools.corpus import preprocessing_eng as preproc
from tools.corpus import corpus_mapping


#CONSTANTS (todo: add this info to config file instead?)
CHUNK_SIZE = 24  #how many tokens per "document"
MINIMUM = 10    #prune terms with frequency <= MINIMUM
IGNORE = ['CC','CD','DT','EX','IN','LS','POS','RP','TO','UH','WDT','WP','WP$','WRB']   #we'll drop any tokens tagged with these parts of speech
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'it', "it's", 'its', 'itself', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def load_dictionary(dict_name):
    return gensim.corpora.Dictionary.load(dict_name)


def get_individual_docs(infile, outfile_path):
    '''Given a corpus, preprocess and save each individual document as its own file. NOTE that in this case a "document" is whatever the situation for the corpus is NOT the chunked version that is used to train the topic model. Assumes infile is in the equitbl schema already (if it isn't, check corpus_mapping)'''
    df = pd.read_json(open(infile))
    
    #preprocess -- DO NOT CHUNK
    lemma_dictionary = preproc.get_pos_lemmas_dictionary(df, ignore_tags = IGNORE, stopwords = STOPWORDS)

    #save them as individual documents
    save_documents(outfile_path, lemma_dictionary)

def get_preprocessed_json(infile, outfile):
    '''Given a corpus, preprocess and save as a different .json file. NOTE that in this case a "document" remains whatever the situation for the corpus; NOT the chunked version used to train a TM. Assumes infile is already in the EQUIBL schema.'''
    df = pd.read_json(open(infile))

    #preprocess -- DO NOT CHUNK
    #TODO: NEED TO DEAL WITH OUT-OF-VOCAB SITUATIONS FOR UNKNOWN TEXTS
    new_df = preproc.convert_pos_lemmas_df(df, ignore_tags = IGNORE, stopwords = STOPWORDS)   

    #save as outfile
    new_df.to_json(outfile, force_ascii=False)

def save_documents(outfile_path, lemma_dictionary):
    ''' support function; saves the individual files '''
    for key, lemmas in lemma_dictionary:
        outfile = open(outfile_path + str(key) + ".txt", "w+")
        for lemma in lemmas:
            lemma = lemma + ' '
            outfile.write(lemma.encode('utf-8'))
        outfile.close()
    

def get_document(filename):
    '''get the words in a document file as a list'''
    file = open(filename, "r")
    document = []
    for line in file:
        line = line.replace('\n', '')
        words = line.split(' ')
        document += words
    file.close()
    return document

def get_topic_matches(bow, model, topic):
    '''Given a bag of words representation of the document; calculates for each topic in the TM: how likely it is, if it were to generate a document of this length, that it would generate the current document '''
    (topics, length) = model.shape #number of topics = number of rows
    logprob = 0.0 #track total words in doc
    length = 0
    for (index, count) in bow: #sum up the logarithms of probabilities
        length += count
        for i in range(count):
            logprob += math.log(model[topic][index])
    return logprob / length

def get_top_documents_from_dir(dictionary_name, document_directory, model_name, topic, total_topics, top_x=100):
    ''' (adapted from main function in get_top_scorers.py) 
    dictionary_name = (incl. path) file where the dictionary is stored
    document_directory = where the documents to be tested are stored
    model_name = (incl. path) file where the _phi.npy part of the TM is stored
    topic = integer representing the topic you're interested in finding the top scorers for
    output_directory = where to store the output
    total_topics = how many topics are in the TM?
    top_x = how many documents should we list? (default: 100)
    '''
    #TODO: MAKE THIS MORE GENERIC (currently only calculates and prints 1 topic)

    dictionary = load_dictionary(dictionary_name)
    model = np.load(model_name)

    files = [f for f in listdir(document_directory) if isfile(join(document_directory, f))]
    print(files) #check
    progress = 0
    scores = [] #will contain: log probabilities and doc_id tuples
    for f in files:
        progress += 1
        if progress % 1000 == 0:
            print(progress)
        doc = get_document(join(document_directory, f))
        bow = dictionary.doc2bow(doc, return_missing=False) #ignore out of vocab words
        lp_avg = get_topic_matches(bow, model, topic)
        (doc_id, ending) = f.split(".")
        #this is purely to double-check that the documents get done right
        print(doc_id)
        print(ending)
        print(f)
        scores.append((lp_avg, doc_id))
    print("Sorting...")
    scores.sort(reverse=True)
    return scores
    #all this function needs to do is to either: write list of doc_ids OR return that info to another function. SCORE info is only really interesting if you're curious about the curve(?  Maybe there's a good cut-off score rather than cut-off subset of corpus? Who knows.)



def save_top_documents_from_dir(dictionary_name, document_directory, model_name, topic, output_directory, total_topics, top_x=100):
    ''' saves the get_top_documents to a file in output_directory '''
    scores = get_top_documents_from_dir(dictionary_name, document_directory, model_name, topic, total_topics, top_x)

    outfile = open(output_directory + 'top_' + str(top_x) + '_t.txt', "w+")
    for i in range(top_x):
        outfile.write(scores[i][1].encode('utf-8') + '\t' + str(scores[i][0]).encode('utf-8') + '\n')
    outfile.close()



def get_top_documents_from_file(dictionary_name, docs_json, model_name, topic, total_topics, top_x=100):
    ''' Does the same thing as get_top_documents_from_dir only assumes that all documents are in the same .json file (with fields 'DOC_ID' and 'TEXT') 
    dictionary_name = (incl. path) file where the dictionary is stored
    docs_json = where the documents to be tested are stored
    model_name = (incl. path) file where the _phi.npy part of the TM is stored
    topic = integer representing the topic you're interested in finding the top scorers for
    output_directory = where to store the output
    total_topics = how many topics are in the TM?
    top_x = how many documents should we list? (default: 100)
    '''
    #TODO: MAKE THIS MORE GENERIC (currently only calculates and prints 1 topic)
    #TODO: make this work for a .json of documents

    dictionary = load_dictionary(dictionary_name)
    model = np.load(model_name)
    docs_df = pd.read_json(open(docs_json)) #makes more sense to pass the df??
    #docs_df = docs_json
    print(docs_df) #check
    progress = 0
    scores = [] #will contain: log probabilities and doc_id tuples
    for i in docs_df.index:
        progress += 1
        if progress % 1000 == 0:
            print(progress)

        doc_id = docs_df[corpus_mapping.ID_FIELD][i]
        doc_string = docs_df[corpus_mapping.TEXT_FIELD][i]
        #doc should be a list, so:
        doc = doc_string.split(" ")
        bow = dictionary.doc2bow(doc, return_missing=False) #ignore out of vocab words
        lp_avg = get_topic_matches(bow, model, topic)
        #this is purely to double-check that the documents get done right
        print(doc_id)
        scores.append((lp_avg, doc_id))
    print("Sorting...")
    scores.sort(reverse=True)
    #TODO: get a graph of the scores? Maybe it's Zipfian?
    #(DO this for our research purposes, at least. it doesn't need to be default behavior...)
    return scores
    #all this function needs to do is to either: write list of doc_ids OR return that info to another function. SCORE info is only really interesting if you're curious about the curve(?  Maybe there's a good cut-off score rather than cut-off subset of corpus? Who knows.)

def save_top_documents_from_file(dictionary_name, doc_json, model_name, topic, output_directory, total_topics, top_x=100):
    ''' saves the get_top_documents to a file in output_directory '''
    scores = get_top_documents_from_file(dictionary_name, doc_json, model_name, topic, total_topics, top_x)

    outfile = open(output_directory + 'top_' + str(top_x) + '_t.txt', "w+")
    for i in range(top_x):
        outfile.write(scores[i][1].encode('utf-8') + '\t' + str(scores[i][0]).encode('utf-8') + '\n')
    outfile.close()
