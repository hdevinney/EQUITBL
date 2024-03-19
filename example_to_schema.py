#!/usr/bin/python3
# -*- coding: utf-8 -*-

# version: 26.04.2022
# author: Hannah Devinney
# EXAMPLE code for transforming a corpus (in this case, nltk's copy of the Brown Corpus) into the .json files
# compatible with the rest of EQUITBL and saving them in the corpora folder.
# Should be quite adaptable to other corpora.

import pandas as pd
import io

#the relevant bits from this package are in tools/corpus/corpus_mapping.py
from tools.corpus.corpus_mapping import map_to_schema

def main():
    out_file_path = 'test_files/input/corpora/'

    #load the brown corpus
    from nltk.corpus import brown
    #the brown corpus is organized by categories
    cats = brown.categories()

    #put corpus into a dataframe
    #for illustrative purposes, we will initially include some information 
    #which we will later get rid of when mapping to the EQUITBL schema
    #but because concat'ing dataframes is terrible, we'll actually build it as a list of dictionaries
    data = []

    #go through category by category:
    for cat in cats:
        f_ids = brown.fileids(cat)
        #go through the category file by file
        for f_id in f_ids:
            #get the text as a list of lists of tokens
            sent_lists = brown.sents(f_id)
            #we just want the text so convert to a string
            sentences = " ".join([token for sentence in sent_lists for token in sentence])
            data.append({'file_id':f_id, 'category':cat, 'sentences':sentences})
    #once you've gone through all of it, convert data into the dataframe:
    full_df = pd.DataFrame(data)

    #save it somewhere
    full_save = out_file_path + 'brown_categories.json'
    full_df.to_json(full_save)
    
    #so now we have a dataframe with the columns 'file_id' 'category' and 'sentences'
    #(and we will end up ignoring the 'category' field
    #but e.g. if you only wanted to convert a subset of the corpus it would be useful!)
    #we have to tell map_to_schema what IDs and TEXTs are currently called:
    id_name = 'file_id'
    text_name = 'sentences' #if you have multiple TEXT columns, list all of them to combine as a single 'article'

    #map to the EQUITBL schema
    new_df = map_to_schema(original_file=full_save, text_path=text_name, id_path=id_name)
    
    #save!
    save_as = out_file_path + 'brown_corpus.json'
    new_df.to_json(save_as)





if __name__ == "__main__":
    main()
