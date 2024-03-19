#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 14.03.2022
# author: Hannah Devinney
# add or remove documents from a corpus

import pandas as pd
import numpy as np
from gensim import corpora
import gensim
import os
from corpus import preprocessing_eng as preproc
from corpus import corpus_mapping
import analyze_documents as analyze

TEXT_FIELD = corpus_mapping.TEXT_FIELD
ID_FIELD = corpus_mapping.ID_FIELD

def new_corpus_by_removal(original, new_name, doc_ids):
    print("remove the specified doc IDs")
    og_df = pd.read_json(open(original))
    #subset of df WITHOUT THE SPECIFIED doc_ids
    new_df = og_df[~og_df.doc_id.isin(doc_ids)]
#    print(new_df) #for double checking number of rows
    #save
    new_df.to_json(new_name, force_ascii=False)
    print("saved to {}".format(new_name))
  

def merge_corpora(corpus_one, corpus_two, new_name):
    #merge two corpora and save under a new name
    new_df = pd.concat([corpus_one, corpus_two], ignore_index=True)
    #(ignore_index actually just resets so indexes aren't repeated during concat)
    new_df.to_json(new_name + ".json")

def get_preproc_corpus(corpora_dir, original_name, new_name):
    #convert to a preprocessed corpus
    original_corpus = corpora_dir + original_name
    new_corpus = corpora_dir + new_name
    print("preprocessing documents...")
    analyze.get_preprocessed_json(original_corpus, new_corpus)
    print("saved to {}".format(new_corpus))
    return new_corpus

def save_top_scores(index, scores, base_dir, k=5000):
    #save the k top-scoring doc_ids (slightly misleading function name oops)
    scores_file = base_dir + "models/doc_given_topic_scores" + str(index) + ".txt"
    i = 0
    print("saving to {}".format(scores_file))
    outfile = open(scores_file, "w+")
    while i < k and i < len(scores):
        score = str(scores[i][1]) #only care about the doc_ids!!
        print("rank: {}\t doc_id: {}".format(i, score))
        outfile.write(score + '\n') 
        i += 1
    outfile.close()

def get_top_scores(index, dictionary, new_corpus, model):
    print("Getting topic {} scores".format(index))
    top_docs = analyze.get_top_documents_from_file(dictionary, new_corpus, model, topic=index, total_topics=15, top_x=5000)
    return top_docs

def load_top_scores(filename):
    #get a saved list of scores
    with open(filename, "r") as f:
        scores = f.readlines()
    return scores
    

