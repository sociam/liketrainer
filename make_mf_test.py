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

# sample random males with replacement

def getRandoMales(rev_id):
	annos = getAnnos(rev_id)
	maleannos = annos.loc[annos['gender'] == 'male']
	maleannossample = maleannos.iloc[np.random.randint(0, len(maleannos), size=10)]
	tox = maleannossample[['toxicity_score']].mean()
	return tox.item()

def getRandoFemales(rev_id):
	annos = getAnnos(rev_id)
	femaleannos = annos.loc[annos['gender'] == 'female']
	femaleannossample = femaleannos.iloc[np.random.randint(0, len(femaleannos), size=10)]
	tox = femaleannossample[['toxicity_score']].mean()
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

print 'relabelling data with random male scores'

# make a new df for male scores
male_test = anncomtest

# make a new column of annotations comprising randomly sampled male annotations with replacement
male_test['male_toxicity_score'] = male_test.rev_id.apply(getRandoMales)

# make a new column saying whether toxic or not according to males
male_test['male_toxicity'] = np.where(male_test['male_toxicity_score']<0, True, False)

# remove newlines and tabs

male_test['comment'] = male_test['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
male_test['comment'] = male_test['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

male_test.to_csv('test_m_detox.csv')


# create new df with random female scores
print 'relabelling data with random female scores'

# make a new df for female scores
female_test = anncomtest

# make a new column of annotations comprising randomly sampled female annotations with replacement

female_test['female_toxicity_score'] = female_test.rev_id.apply(getRandoFemales)

# make a new column saying whether toxic or not according to females

female_test['female_toxicity'] = np.where(female_test['female_toxicity_score']<0, True, False)

female_test['comment'] = female_test['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
female_test['comment'] = female_test['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

female_test.to_csv('test_f_detox.csv')