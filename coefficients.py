import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os
import glob
import sys  
import cPickle
import pickle
import numpy as np

coeffdf = pd.DataFrame()

models = glob.glob('models/*.sav')

for model in models:
	pickled_model = open(model, 'rb')
	clf = pickle.load(pickled_model)
	ngrams = clf.steps[0][1].get_feature_names()
	strengths = clf.steps[2][1].coef_[0]
	ngrams_col_name = model + '_ngrams'
	strengths_col_name = model + '_strengths'
	coeffdf[ngrams_col_name] = ngrams
	coeffdf[strengths_col_name] = strengths
