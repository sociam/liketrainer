import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os
import glob
import sys  
import cPickle
import numpy as np

print 'Making test data....'

# load up the training data and demographics of workers
comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t')
annodemog = pd.read_csv('annodemog_mixlabels.csv', sep = '\,')

# find annotators for a rev_id

def getAnnos(rev_id):
	return annodemog.loc[annodemog['rev_id'] == rev_id]

# sample random balanced males females with replacement

def getRandoMix(rev_id):
	annos = getAnnos(rev_id)
	femaleannos = annos.loc[annos['gender'] == 'female']
	maleannos = annos.loc[annos['gender'] == 'male']
	femaleannossample = femaleannos.iloc[np.random.randint(0, len(femaleannos), size=5)]
	maleannossample = maleannos.iloc[np.random.randint(0, len(maleannos), size=5)]
	f = femaleannossample['toxicity_score'].values
	m = maleannossample['toxicity_score'].values
	mixedannossample = np.concatenate((m, f), axis=0)
	tox = mixedannossample.mean()
	return tox.item()


# select only rows where mixed_gender is true

annodemogmix = annodemog[annodemog.mixed_gender]

# merge annodemog with comments

anndemcom = pd.merge(annodemogmix, comments)

# remove duplicate comments

anncom = anndemcom.drop_duplicates('rev_id')

# remove all training data, leave test data

anncomtest = anncom.query("split=='test'")

# create new df with random males scores

print 'relabelling data with random mixed scores'

# make a new df for mixed scores
mixed_test = anncomtest

# make a new column of annotations comprising randomly sampled male annotations with replacement
mixed_test['mixed_toxicity_score'] = mixed_test.rev_id.apply(getRandoMix)

# make a new column saying whether toxic or not according to males
mixed_test['mixed_toxicity'] = np.where(mixed_test['mixed_toxicity_score']<0, True, False)

# remove newlines and tabs

mixed_test['comment'] = mixed_test['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
mixed_test['comment'] = mixed_test['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

mixed_test.to_csv('test_mixed_detox.csv')