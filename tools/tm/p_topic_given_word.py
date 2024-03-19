#!/usr/bin/python3
# -*- coding: utf-8 -*-

#N/B copied the functions into model_printing to make the exclusivity color-coding be all in one place

import gensim
from gensim import corpora
import numpy as np
from argparse import ArgumentParser
import math

def load_dictionary(name):
    return gensim.corpora.Dictionary.load(name)

def get_min(list):
    minval = 1.0
    pos = 0
    for i in range(len(list)):
        (val,idx) = list[i]
        if val < minval:
            minval = val
            pos = i
    return (minval, pos)

def get_top_words(t, phi, dict, k):
    k_best = []
    topic = phi[t]
    for i in range(k):
        k_best.append((topic[k],i))
    (minval, minpos) = get_min(k_best)
    for i in range(k,len(topic)):
        if (topic[i] > minval):
            k_best[minpos] = (topic[i], i)
            (minval, minpos) = get_min(k_best)
    k_best.sort(reverse = True)
    #for (val, pos) in k_best:
    #    print(dict.get(pos).encode('utf-8'))
    #    print("\t\t %.5f\n" % (val))
    return k_best

def get_topic_probabilities(topic, k_best, phi, dict):
    (k, n) = phi.shape
    topic_probs = []
    for (val, pos) in k_best:
        #print(val)
        #print(pos)
        #print(dict.get(pos))
        #print(phi[topic][pos])
        topic_prob = phi[topic][pos]
        prob_sum = 0.0
        for i in range(k):
            prob_sum += phi[i][pos]
        #print(topic_prob / prob_sum)
        #print()
        topic_probs.append(topic_prob / prob_sum)
            
    return topic_probs
        
def main():
    a = ArgumentParser()
    a.add_argument('-dict', dest='dictionary_name', required=True, type=str, help="saved dictionary file")
    a.add_argument('-model', dest='model_name', required=True, type=str, help="the topic model to use")
    a.add_argument('-outdir', dest='output_dir', required=True, type=str, help="where results are saved")
    a.add_argument('-name', dest='file_base', required=True, type=str, help="what to call the output file")
    a.add_argument('-csv', dest='use_csv', required=False, default=False, type=bool, help="use this flag to get a csv instead of tab-separated txt file")
    opts = a.parse_args()
    print("Loading dictionary " + opts.dictionary_name)
    dict = load_dictionary(opts.dictionary_name)
    
    print("Loading model " + opts.model_name)
    model = np.load(opts.model_name)

    for topic in range(3):
        k_best = get_top_words(topic, model, dict, 50)
        t_probabilities = get_topic_probabilities(topic, k_best, model, dict)
        print(t_probabilities)
        if(opts.use_csv):
            outfile_name = opts.output_dir + "p_topic_given_word_" + opts.file_base + str(topic) + ".csv"
        else:
            outfile_name = opts.output_dir + "p_topic_given_word_" + opts.file_base + str(topic) + ".txt"

        outfile = open(outfile_name, "w+")

        if(opts.use_csv):
            outfile.write("WORD,WEIGHT_INT_TOPIC,RELATIVE_WEIGHT\n")
            for i in range(50):
                (val, pos) = k_best[i]
                outfile.write(dict.get(pos).encode('utf8')) #ensure Swedish prints
                outfile.write(",%.5f" % (val))
                outfile.write(",%.5f\n" % (t_probabilities[i]))
        else:
            for i in range(50):
                (val, pos) = k_best[i]
                outfile.write(dict.get(pos).encode('utf8'))
                outfile.write("\t\t %.5f" % (val))
                outfile.write("\t\t %.5f\n" % (t_probabilities[i]))
        outfile.close()
    
    
    
if __name__ == "__main__":
    main()
