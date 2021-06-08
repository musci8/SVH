# SVH
Code for extracting Statistically Validated Hypergraphs

Based on
F. Musciotto, F. Battiston, R. N. Mantegna, Detecting informative higher-order interactions in statistically validated hypergraphs, arxiv.org/abs/2103.16484

## Setup

```pip install -r requirements.txt```

The computation of p-values is made through the 'SuperExactTest' package in R (which is called through ```rpy2```). To install it, you need to open R and type ```install.packages('SuperExactTest');```. Next versions of the repo will include native Python code to compute the p-values.

## Usage

The Jupyter notebook Example.ipynb provides examples to use the code and replicate some of the results of arxiv.org/abs/2103.16484

## Data

In the ```data/``` folder you find all data used in arxiv.org/abs/2103.16484
