# liketrainer
Code for the 'Like Trainer, Like Bot' study, presented at [SocInfo 2017](socinfo2017.oii.ox.ac.uk)

## Training data

Data used in the study is taken from previous work by [Wulczyn et al](https://arxiv.org/abs/1610.08914), and can be found [here](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release)

## Building classifiers

The basic classifier (using all the training data) is built with `make_clf.py`. Male-only, female-only and mixed-gender classifiers are labelled accordingly.

`make_models.py` builds 10 classifiers. In order to generate random samples that are reproducible, the numpy random seed function is used. The resulting classifiers are named 1-10 after the random seed used to generate the sample on which they were trained.

`coefficients.py` extracts the coefficients from a set of classifiers.

## Building test data

The test dataset used is `test_detox.csv` and is generated with `make_mixed_test.py`.

## Results

The results of the main tests are in `test_results_balanced.csv`.
