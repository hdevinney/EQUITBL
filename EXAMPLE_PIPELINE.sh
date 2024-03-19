#!/bin/bash

#shortcut to tm folder
TM=`pwd`'/tools/tm'

#step 1: set up your python environment (replace with whatever path is relevant for you)
ENVIRONMENT='/path/to/<your_environment>/bin/activate'
source $ENVIRONMENT

#step 2: if your corpus is not already in the right format, convert it
python3 example_to_schema.py

#step 2.5: sort out your CONFIG and LOG files
CONFIG='config_files/example_config.ini' #make sure to go into this and set appropriate absolute paths!
LOG='logs/log_EXAMPLE.txt'

#step 3 if your corpus is not already preprocessed, do it now
#N/B: replace with appropriate preprocessing python files as necessary. This one gets the lemmas and parts of speech tags, and then splits articles into documents of 24 terms (moving window)
python3 example_preprocess.py -config $CONFIG

#(if you have a memory limit increase here is where to do it)
ulimit -d unlimited -v unlimited

#step 4: train the model (+collect timing info)
#/bin/time -o $LOG -v python3 $TM/run_and_save_tm.py -config $CONFIG #OPTIONAL log the time it takes to train
python3 $TM/run_and_save_tm.py -config $CONFIG

#step 5: get the VISUALIZATIONS of that model
OUTNAME='brown_example'
SEEDS='gendered_lemmaPOS/all_seeds.txt'
TITLE='BROWN_CORPUS'
#python3 visualize_topics.py -config $CONFIG -outname $OUTNAME -seeds $SEEDS -k 30 -num_topics 3 -title $TITLE

