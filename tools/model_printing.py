#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 2.0 (03.01.2022)
# author: Hannah Devinney (with some code taken and/or adapted from Henrik Björklund)
# toolkit for pretty printing and logging topic modeling output. Also visualizing topics. 

import os
import matplotlib.pyplot as plt
import matplotlib.axes

def get_min(list):
    '''Support function for printing best topic stuff'''
    minval = 1.0
    pos = 0
    for i in range(len(list)):
        (val,dix) = list[i]
        if val < minval:
            minval = val
            pos = i
    return (minval, pos)

def color_by_exclusivity(value):
    '''Support function for visualizing topics.'''
    if value < 0.5:
        return 'white'
    elif 0.5 <= value < 0.75:
        return 'blue'
    elif 0.75 <= value < 0.9:
        return 'magenta'
    else: #value >= 0.9
        return 'red'

def get_k_best(topic, k):
    '''Returns k best tuples of (value, position) where value=topic weight and position=id in dictionary '''
    k_best = []
    for i in range(k):
        k_best.append((topic[i],i))
#    print(k_best)
    (minval, minpos) = get_min(k_best)
    for i in range(k, len(topic)):
        if (topic[i] > minval):
            k_best[minpos] = (topic[i], i)
            (minval, minpos) = get_min(k_best)
    k_best.sort(reverse=True) #remove this to flip the charts?
    return k_best

def get_topic_probabilities(topic, k_best, phi):
    '''Returns relative weights p(t|w) for the k_best terms in a model.'''
    (k, n) = phi.shape
    topic_probs = []
    for (val, pos) in k_best:
#        print(val)
#        print(pos)
        topic_prob = phi[topic][pos]
        prob_sum = 0.0
        for i in range(k):
            prob_sum += phi[i][pos]
        topic_probs.append(topic_prob / prob_sum)
    return topic_probs

def make_chart(k, terms, weights, colors, output_file=None, show_prob=False, chart_name=None, logscale=True, legend=False):
    '''Support function for visualize_topic() (construct the chart) '''
    #set up chart aesthetics.
    line_weight = [1]*k
    bar_width = 0.9
    xmin = 0.001
    xmax = 0.15
    w, h = plt.figaspect(0.75)
    plt.figure(figsize=(w,h))
    #viz
    
    bars = plt.barh(terms, weights, color=colors,
             height=bar_width, linewidth=line_weight, edgecolor='black')
    plt.xlim(xmin, xmax)
    plt.xlabel("Probability")
    plt.ylabel("Term")
    if logscale is True:
        plt.xscale('log')
    if chart_name is not None:
        plt.title(chart_name)
    if show_prob:
        x = 1 #x position
        for bar in bars:
            width = bar.get_width() #bar value
            y = bar.get_y() # y position
            plt.text(x, y, s=f'{width:.5f}')
    #add a legend
    if legend:
        color_map = {'rel >= 0.9':'red', '0.9 > rel >= 0.75':'magenta', '0.75 > rel >= 0.5':'blue', '0.5 > rel':'white'}
        labels = list(color_map.keys())
        handles = [plt.Rectangle((0,0),1,1, facecolor=color_map[label], edgecolor='black') for label in labels]
        #place the legend at the bottom right (outside chart area)
        plt.legend(handles, labels, bbox_to_anchor=(1,0), loc='upper left')
    #fix margins
    plt.tight_layout()
    #save?
    if output_file is not None:
        out = output_file + ".pdf"
        plt.savefig(out, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def get_terms_weights(k_best, dict, seed_terms):
    '''Support function for visualize_topic (include all terms, and adds a * before seed terms)'''
    weights = []
    terms = []
    for (val, pos) in k_best:
        weights.append(val)
        term = dict.get(pos)
        if term in seed_terms:
            term = "*" + term
        terms.append(term)
    return weights, terms

def get_seedless_terms_weights(k_best, dict, seed_terms):
    '''Support function for visualize_topic (ignore ALL seed terms - not just the ones for this topic!)'''
    weights = []
    terms = []
    new_k_best = []
    for (val, pos) in k_best:
        term = dict.get(pos)
        if term not in seed_terms:
            weights.append(val)
            terms.append(term)
            new_k_best.append((val, pos))
    return weights, terms, new_k_best
    

def visualize_topic(topic_index, dict, k, phi, seed_terms, show_prob=False, chart_name=None, p_tw=False, output_file=None, logscale=True, show_seeds=True):
    '''Displays the k best words (and their probabilities) in a barchart. Set show_prob to True to print exact values. If you want to label the chart, assign a string to chart_name. Set p_tw to true to include color-coded relative weights (False to have every bar be the same color). Save to output_file (if None, just displays the chart). Set logscale=False to get a chart with linear values.'''
    topic = phi[topic_index]
    k_best = get_k_best(topic, k)
    print(k_best)
    k_best.reverse() #'flip' so highest on top


    if show_seeds:
        weights, terms = get_terms_weights(k_best, dict, seed_terms)
    else:
        weights, terms, k_best = get_seedless_terms_weights(k_best, dict, seed_terms)
        #replace k because we might now have fewer terms to deal with
        k = len(weights)
       

    #deal with relative weights/exclusivity
    if p_tw: #color-coding
        colors = []
        tw = get_topic_probabilities(topic_index, k_best, phi)
        for i in range(k):
            colors.append(color_by_exclusivity(tw[i]))
        legend = True
    else: #no fill
        colors = ['white'] * k
        legend = False

    #do the actual vizualization
    make_chart(k, terms, weights, colors, output_file=output_file, show_prob=show_prob, chart_name=chart_name, logscale=logscale, legend=legend)





    
def print_topic(topic, dict, k):
    '''Prints the k best words (and their probabilities) in topic '''
    k_best = get_k_best(topic, k)
    for (val, pos) in k_best:
        print(dict.get(pos) + ' ' + str(val) + ' ')

def print_topics(phi, dict, k):
    '''Prints the k best words (and their probabilities) for all topics in phi.'''
    i = 1
    for topic in phi:
        print('\nTopic ' + str(i) + ':')
        print_topic(topic, dict, k)
        i += 1

def print_topic_to_file(topic, dict, k, outfile):
    '''Prints the k best words (and their probabilities) in topic; saves to specified outfile stream.'''
    k_best = get_k_best(topic, k)
    for (val, pos) in k_best:
        outfile.write(dict.get(pos))
        outfile.write("\t\t %.5f\n" %(val))

def print_to_file(phi, dict, k, outfilename):
    '''Prints the k best words (and their probabilities) in all topic in phi; saves them to a file located at outfilename'''
    outfile = open(outfilename, "w")
    i = 1
    for topic in phi:
        outfile.write('Topic ' + str(i) + ":\n")
        print_topic_to_file(topic, dict, k, outfile)
        outfile.write("\n")
        i += 1
    outfile.close()

def print_topic_to_file(topic, dict, k, outfile):
    '''Prints the k best words (and their probabilities) in topic; saves to specified outfile stream.'''
    k_best = get_k_best(topic, k)
    for (val, pos) in k_best:
        outfile.write(dict.get(pos))
        outfile.write("\t\t %.5f\n" %(val))

def print_seedless_topic_to_file(topic, dict, k, seeds, outfile):
    '''Prints the k best words (and their probabilities) in topic; saves to specified outfile stream.'''
    k_best = get_k_best(topic, k)
    for (val, pos) in k_best:
        if dict.get(pos) not in seeds: #check if a seed word; ignore if so
            outfile.write(dict.get(pos))
            outfile.write("\t\t %.5f\n" %(val))

        
def print_to_file_seedless(phi, dict, k, seeds, outfilename):
    '''Prints the k best words (and their probabilities), EXCEPT for terms in seed word lists, in all topic in phi; saves them to a file located at outfilename. (NB this means you will end up with between k-(seeds length) and k terms)'''
    outfile = open(outfilename, "w")
    i = 1
    all_seeds = []
    #check if seeds is a list of lists
    if any(isinstance(elem, list) for elem in seeds):
        #flatten
        all_seeds = [item in elem for elem in seeds]
    else: #already flat
        all_seeds = seeds

    print("ignoring terms: {}".format(all_seeds))
    
    for topic in phi:
        outfile.write('Topic ' + str(i) + ":\n")
        print_seedless_topic_to_file(topic, dict, k, all_seeds, outfile)
        outfile.write("\n")
        i += 1
    outfile.close()


#this one is just kinda a template for a generic log, modify as needed...
def log_counts(logfile, counts, estphi, esttheta):
    '''Logs some basic data about the model to specified logfile.'''
    outfile = open(logfile, "a")
    tmp = 0
    for count in counts:
        outfile.write("count for topic {} is : \t{}\n".format(tmp, count))
        tmp += 1
    #this info is largely uninteresting for most users
    #outfile.write("\nestimated phi length: " + str(estphi))
    #outfile.write("\nestimated theta length: " + str(esttheta))

