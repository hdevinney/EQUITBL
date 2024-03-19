#!/usr/bin/python3

# version: 1.0 (12.04.2021)
# author: Hannah Devinney
# experimental tf-idf code
# N/B: trying to figure out at what point it makes sense to pass things to C/C++


import numpy as np
import math
from gensim import corpora

def get_doc_freq_vector(dictionary):
    '''Takes a gensim dictionary
    Returns a list of doc_freqs: How many documents does each term appear in?'''
    vocab_size = len(dictionary)
    #token ids become index/location in list
    doc_freq_vec = [dictionary.dfs[i] for i in range(vocab_size)]
    return doc_freq_vec

#takes a gensim dictionary and a corpus (BoW)
#returns a 2d array of the term-x-document matrix
def get_tf_matrix(dictionary, corpus):
    '''Takes a gensim dictionary and a BoW corpus
    Returns a 2d array of the term-by-document matrix'''
    
    #initialize term-x-document matrix: rows=terms, cols=docs
    tf_mat = np.zeros(shape=(len(dictionary),dictionary.num_docs), dtype=int)

    for i in range(dictionary.num_docs):
        #get term frequencies for document i:
        term_freqs = corpus[i]
        #get tf array for this document
        #TODO: find a way to do it without nesting for-loops??
        tmp_tf = np.zeros(shape=(len(dictionary)), dtype=int)
        for token_id, freq in term_freqs:
            tmp_tf[token_id] = freq
        tf_mat[:,i] = tmp_tf

    return tf_mat

def get_length_adjusted_tf(raw_tf, dictionary):
    '''Takes a matrix of raw term frequences and a gensim dictionary
    returns the length-adjusted term frequencies: tf = freq(term,doc)/length(doc)'''

    #tf = (freq(i,j)/doc_len(j))
    tf = np.zeros(shape=(len(dictionary),dictionary.num_docs))
    #total words in each document
    doc_len = np.sum(raw_tf, axis=0)
    for i in range(len(dictionary)):
        for j in range(dictionary.num_docs):
            tf[i,j] = raw_tf[i,j] / doc_len[j]
    return tf

def get_augmented_freq(raw_tf, dictionary):
    '''Takes a matrix of raw term frequences and a gensim dictionary
    returns the augmented term frequencies: tf = 0.5+((0.5*freq(term,doc))/max_freq_in(doc)'''


    #tf = (0.5 + ((0.5(freq(i,j)))/max(j)))
    tf = np.zeros(shape=(len(dictionary),dictionary.num_docs))
    #frequency of most frequent word per document
    maxjs = np.nanmax(raw_tf, axis=0)
    for i in range(len(dictionary)):
        for j in range(dictionary.num_docs):
            tf[i,j] = (0.5 + ( (0.5*raw_tf[i,j]) / maxjs[j] ) )
    return tf
    
 
    
#calculate tf-idf (default: raw-count tf)
def tf_idf(tf_mat, doc_freq, dictionary, tf_eq="raw_freq"):
    '''Takes matrix of raw term frequences, a document frequency vector, and a gensim dictionary
    Calculates the term-frequency inverse-document-frequency
    Optional: set which term-freqeuncy calculation to use with tf_eq from ("raw_freq", "length_adjusted", "augmented_freq")'''

    #throughout this function: i = term_idx, j = doc_idx 

    #calculate tf(i,j)
    if tf_eq is "raw_freq": #default
        #tf_mat already contains the right weights
        tf = tf_mat
    elif tf_eq is "length_adjusted":
        tf = get_length_adjusted_tf(tf_mat, dictionary)
    elif tf_eq is "augmented_freq":
        tf = get_augmented_freq(tf_mat, dictionary)
    else:
        print("Invalid tf_eq")

#    print("tf is {}".format(tf))

    #I'm about 85% sure natural log is the correct move so...
    idf = [math.log((dictionary.num_docs/df)) for df in doc_freq]
#    print("idf is {}".format(idf))

    tfidf = np.zeros(shape=(len(dictionary),dictionary.num_docs))

    #tfidf(i,j) = tf(i,j) x idf(i)
    for i in range(len(dictionary)):
        for j in range(dictionary.num_docs):
            tfidf[i,j] = tf[i,j] * idf[i]

#    print("TFIDF RESULT:")
#    print(tfidf)
    return tfidf
    
        


def main():
    #MAIN IS ONLY FOR TESTING PURPROSES, REALLY...

    #TINY test
    test_corpus_sents = ["a sentence", "another sentence", "still another sentence still"]
    #bigger/mwe 
#    test_corpus_sents = ["Same tax issues now for same-sex couples.", "Gay and lesbian couples got relationship equality with the June 26 U.S. Supreme Court ruling that the Defense of Marriage Act, or DOMA, is unconstitutional.", "Now that the law has been struck down, same-sex marriage is legal in the eyes of the federal government.", "That means that same-sex married couples can, among other tax matters, now file joint tax returns.", " But as is often the case with taxes, such equality may work in their favor."]
    
    #preprocessing
    #(here for test simplicity: just splitting and lowercasing...)
    test_texts = [[word for word in sent.lower().split()] for sent in test_corpus_sents]
    

    #get: dictionary
    dictionary = corpora.Dictionary(test_texts)

    #get: number of documents (should come out of dictionary/corpus..?)
    num_docs = dictionary.num_docs
    print("There are {} documents".format(num_docs))

    #get doc_freq vector (for each TERM, how many documents does it appear in?)
    doc_freq_vec = get_doc_freq_vector(dictionary)
    print("The document-frequency vector: {}".format(doc_freq_vec))

    corpus = [dictionary.doc2bow(text) for text in test_texts] #list of list of tuples: (id, freq) 

    #get: tf_matrix (for each document, how many times does it contain each term?)
    tf_matrix = get_tf_matrix(dictionary, corpus)
    print("The term-frequency matrix: {}".format(tf_matrix))

    #update weights in term_doc matrix (from tf to tf-idf) via multiplication
#    tfidf = tf_idf(tf_matrix, doc_freq_vec, dictionary)
    tfidf = tf_idf(tf_matrix, doc_freq_vec, dictionary, tf_eq="augmented_freq")



if __name__ == "__main__":
    main()
