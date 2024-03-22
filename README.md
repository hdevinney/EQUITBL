# EQUITBL
Explore, Query, &amp; Understand Implicit Textual Bias in Language data

The EQUITBL package provides support for a mixed-methods approach to analyzing bias in corpora using semi-supervised topic modeling. This approach involves using a quantitative method for filtering relevant information out of a corpus, and then analyzing the results with qualitative methods.

Topic modeling allows us to identify underlying patterns in a corpus, and semi-supervised variants let us find patterns (such as themes and closely-associated documents) specific to particular groups.

The a more thorough user guide is available here (`EQUITBL_instructions.pdf`), which includes details for the qualitative analysis parts of the method.

## Dependencies

* [pSSLDA](https://github.com/davidandrzej/pSSLDA)
* [gensim](https://radimrehurek.com/gensim/)
* numpy
* pandas
* maptlotlib (only for visualizing results)
* nltk (if you would like to run the example code)

## Running the example code

You will have to make a few minor changes to sucessfully run the example code:

* update the absolute paths in config_files/example_config.ini for project_root (should be /path/to/equitbl/test_files) and log_file (should be /path/to/equitbl/logs/<desired_file>.txt)
* update the value for ENVIRONMENT in EXAMPLE_PIPELINE.sh to the path to <your_virtual_environment>/bin/activate

Once these changes are made, you should be able to simply run ./EXAMPLE_PIPELINE.sh from the equitbl directory in order to confirm everything is in working order!

## Troubleshooting

Some things in the code remain non-optimal. Here are some things to try.

* If the import for model_printing in run_and_save_tm.py is not playing nicely:
  - Replace the line `sys.path.append('..')` with the following lines, which should do about the same thing:
    ```
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    ```
  - Alternately, you can just add the absolute path string with sys.path.append()

## Credits

This package is co-written with [Henrik Björklund](https://github.com/henrikb-umu).

### Citation information

If you use the EQUITBL package in your work, please cite the following:

Hannah Devinney, Jenny Björklund, and Henrik Björklund (2020). [Semi-supervised topic modeling for gender bias discovery in English and Swedish](https://aclanthology.org/2020.gebnlp-1.8/). In _Proceedings of the Second Workshop on Gender Bias in Natural Language Processing_, pages 79-82.

We also have some works demonstrating some uses for EQUITBL in identifying gendered patterns across news and web corpora:

Hannah Devinney, Jenny Björklund, and Henrik Björklund (2020). [Crime and Relationship: Exploring gender bias in NLP corpora](https://spraakbanken.gu.se/sites/default/files/2022/SLTC-2020_paper_10.pdf). In _The Eigth Swedish Language Technology Conference_ (SLTC)

### Acknowledgements


Thanks to Igor and Sal for recording and reporting all the issues they found when setting up from scratch!


