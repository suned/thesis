# Deep Multi-Task Learning For Relation Extraction

Master thesis by Sune Debel, written in the spring of 2017.

Project plan can be found [here](https://app.teamweek.com/#p/dwtqvjq2igcforiqfiex).

# Report
## Dependencies
- [texlive](https://www.tug.org/texlive/)
- [ku-forside](http://www.math.ku.dk/~m00cha/) (Place in texmf/tex folder).

## Build Instructions
To generate report do:

    > cd report; make

The latest built version can be found [here](https://github.com/suned/thesis/raw/master/report/sune_debel_master_thesis.pdf).


# Experiment
## Dependencies
 - [theano](http://deeplearning.net/software/theano/install.html)
 
 The model arcitechture currently prevents the use of tensorflow.

## Run Instructions
Install requirements:

    > pip install -r experiment/requirements.txt

If using a virtual environment (recommended), be sure to install in the same
environment as `theano` is installed.

Download `spacy` models:

    > python -m spacy download en
 
Run:

    > cd experiment
    > python -m run -h
