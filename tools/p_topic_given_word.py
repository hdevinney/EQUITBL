import gensim
from gensim import corpora
import numpy as np
from argparse import ArgumentParser
import math
import model_printing as mp #for convenience, get seedless topic lists at the same time

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
    positions = []
    topic = phi[t]
    for i in range(k):
        k_best.append((topic[k],i))
        positions.append(i)
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


def print_to_txt(outfile_name, dict, k, k_best, t_probabilities):
    outfile_name = outfile_name + ".txt"
    outfile = open(outfile_name, "w+", encoding='utf8')
    for i in range(k):
        (val, pos) = k_best[i]
        outfile.write(dict.get(pos))
        outfile.write("\t\t %.5f" % (val))
        outfile.write("\t\t %.5f\n" % (t_probabilities[i]))
    outfile.close()

def print_to_txt_seedless(outfile_name, dict, k, k_best, t_probabilities, seeds):
    outfile_name = outfile_name + ".txt"
    outfile = open(outfile_name, "w+", encoding='utf8')
    for i in range(k):
        (val, pos) = k_best[i]
        if dict.get(pos) not in seeds: #check if a seed word; ignore if so
            outfile.write(dict.get(pos))
            outfile.write("\t\t %.5f" % (val))
            outfile.write("\t\t %.5f\n" % (t_probabilities[i]))
    outfile.close()

def print_to_csv(outfile_name, dict, k, k_best, t_probabilities):
    outfile_name = outfile_name + ".csv"
    outfile = open(outfile_name, "w+", encoding='utf8')
    outfile.write("WORD,WEIGHT_INT_TOPIC,RELATIVE_WEIGHT\n")
    for i in range(k):
        (val, pos) = k_best[i]
        outfile.write(dict.get(pos))
        outfile.write(",%.5f" % (val))
        outfile.write(",%.5f\n" % (t_probabilities[i]))
    outfile.close()

def print_to_csv_seedless(outfile_name, dict, k, k_best, t_probabilities, seeds):
    outfile_name = outfile_name + ".csv"
    outfile = open(outfile_name, "w+", encoding='utf8')
    outfile.write("WORD,WEIGHT_INT_TOPIC,RELATIVE_WEIGHT\n")
    for i in range(k):
        (val, pos) = k_best[i]
        if dict.get(pos) not in seeds: #check if a seed word; ignore if so
            #print(dict.get(pos))
            outfile.write(dict.get(pos))
            outfile.write(",%.5f" % (val))
            outfile.write(",%.5f\n" % (t_probabilities[i]))
    outfile.close()


  

def main():
    a = ArgumentParser()
    a.add_argument('-dict', dest='dictionary_name', required=True, type=str, help="saved dictionary file")
    a.add_argument('-model', dest='model_name', required=True, type=str, help="the topic model to use (_phi)")
    a.add_argument('-topics', dest = 'num_topics', required=True, type=int, help="number of topics to be calculated")
    a.add_argument('-k', dest='k', required=True, type=int, help="number of terms in a topic to include")
    a.add_argument('-outdir', dest='output_dir', required=True, type=str, help="where results are saved")
    a.add_argument('-name', dest='file_base', required=True, type=str, help="what to call the output file")
    a.add_argument('-csv', dest='use_csv', required=False, default=False, type=bool, help="use this flag to get a csv instead of tab-separated txt file")
    a.add_argument('-seedless', dest='print_seedless', required=False, type=bool, help="use this flag to hide seed terms from results")
    a.add_argument('-seeds', dest='seeds', required=False, type=str, help="REQUIRED IF SEEDLESS: file containing all seed terms")
    
    opts = a.parse_args()
    print("Loading dictionary " + opts.dictionary_name)
    dict = load_dictionary(opts.dictionary_name)
    
    print("Loading model " + opts.model_name)
    model = np.load(opts.model_name)

    for topic in range(opts.num_topics):
        k_best = get_top_words(topic, model, dict, opts.k)
        t_probabilities = get_topic_probabilities(topic, k_best, model, dict)
        print(t_probabilities)

        outfile_name = opts.output_dir + "p_topic_given_word_" + opts.file_base + str(topic)
        print("saving results to {}".format(outfile_name))

        if(opts.print_seedless): #only print terms not in seed list
            #get the seed terms into a list
            seeds = []
            list_file = open(opts.seeds)
            for line in list_file:
                line = line.replace('\n', '')
                seeds.append(line)
            list_file.close()
            print("seeds to be ignored: {}".format(seeds))
            #don't need to turn them into indexes, I'm pretty sure...
            if(opts.use_csv):
                print_to_csv_seedless(outfile_name, dict, opts.k, k_best, t_probabilities, seeds)
            else:
                print_to_txt_seedless(outfile_name, dict, opts.k, k_best, t_probabilities, seeds)

                
                
        else: #print all k terms
            if(opts.use_csv):
                print_to_csv(outfile_name, dict, opts.k, k_best, t_probabilities)
            else:
                print_to_txt(outfile_name, dict, opts.k, k_best, t_probabilities)


    
    
    
if __name__ == "__main__":
    main()
