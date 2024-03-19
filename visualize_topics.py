#!/usr/bin/python3
# -*- coding: utf-8 -*-

#version 1 (01.03.2022)
#author: Hannah Devinney
#visualize topic(s) from a model
#uses the config file for creating a topic model

import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
import tools.model_printing as mp
import tools.tm.p_topic_given_word as ptw
import gensim
import numpy as np


def main():
    ##################### DEAL WITH ARGUMENTS  #######################
    a = ArgumentParser()
    a.add_argument('-config', dest='config_file', required=True, type=str, help="path to .ini configuration file")
    a.add_argument('-outname', dest='out_name', required=True, type=str, help="what you would like the .pdf files to start with (e.g. baseline_model_ will produce a visualization for topic 1 called baseline_model_1.pdf")
    a.add_argument('-seeds', dest = 'seeds', required=True, type=str, help="which all_seeds.txt file do you want to read the seed terms from? (base: <project_root>/intput/seeds/)")
    a.add_argument('-k', dest='k', default=30, type=int, help="how many terms (maximum) should appear in the visualization?")
    a.add_argument('-num_topics', dest='num_topics', type=int, help="create visuals for the first num_topics topics in a model. If NULL, visualizes every topic.")
    a.add_argument('-title', dest='title_base', default="", help="title of each visualization will be '<title> topic X'")
    a.add_argument('-show_prob', dest='show_prob', default=True, type=bool, help="set to true to display exact probability values (False: no values diplayed)")
    a.add_argument('-colors', dest='p_tw', default=True, type=bool, help="set to true to color-code by p(t|w) 'exclusivity' score (False: all bars will be white)")
    a.add_argument('-logscale', dest='logscale', default=True, type=bool, help="set to true to display x-axis on a log scale (False: linear scale)")
    a.add_argument('-show_seeds', dest='show_seeds', default=False, type=bool, help="set to True to display seeds (False (default): hide all seed terms)")

    opts = a.parse_args()
    config = ConfigParser()
    config.read(opts.config_file)

    BASE_DIRECTORY = config['FILE_PATHS']['project_root']
    DICTIONARY_NAME = config['FILE_PATHS']['dictionary_name']
    DICT = BASE_DIRECTORY + 'models/bow/' + DICTIONARY_NAME + '.gensim'
    WORD_LIST_DIRECTORY = BASE_DIRECTORY + "input/seeds/"
    OUT_BASE = BASE_DIRECTORY + "output/vis/" + opts.out_name
    MODEL_NAME = config['FILE_PATHS']['output_name'] 
    MODEL_FILE = BASE_DIRECTORY + "models/tm/" + MODEL_NAME + "_phi.npy"

    ################################################################

    #load topics, dictionary, seed terms
    dictionary = gensim.corpora.Dictionary.load(DICT)
    model = np.load(MODEL_FILE)
    seed_terms = []
    list_file = open(WORD_LIST_DIRECTORY + opts.seeds)
    for line in list_file:
        line = line.replace('\n', '')
        seed_terms.append(line)
    list_file.close()

    #settings
    k = opts.k
    if opts.num_topics:
        #visualize each of the first num_topics topics
        for i in range(0, opts.num_topics):
            name = opts.title_base + " topic " + str(i+1) #indexing weird sorry
            OUT = OUT_BASE + str(i)
            mp.visualize_topic(topic_index=i, dict=dictionary, k=k, phi=model, seed_terms=seed_terms, show_prob=opts.show_prob, chart_name=name, p_tw=opts.p_tw, output_file=OUT, logscale=opts.logscale, show_seeds=opts.show_seeds)
    else: #get all the topics
        print("TODO: make sure this actually works correctly")
        for i in range(0, len(model)):
            topic = model[i]
            name = opts.title_base + " topic " + str(i+1) #indexing weird sorry
            OUT = OUT_BASE + str(i)
            mp.visualize_topic(topic_index=i, dict=dictionary, k=k, phi=model, seed_terms=seed_terms, show_prob=opts.show_prob, chart_name=name, p_tw=opts.p_tw, output_file=OUT, logscale=opts.logscale, show_seeds=opts.show_seeds)




if __name__ == "__main__":
    main()
