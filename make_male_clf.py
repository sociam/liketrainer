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

seed = sys.argv[1]
np.random.seed(int(seed))

print 'Making male-only classifier...'

# load up the training data and demographics of workers
comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t')
annodemog = pd.read_csv('annodemog_mixlabels.csv', sep = '\,')
test_detox = pd.read_csv('test_detox.csv')
imper = pd.read_csv('datasets/impermium_offence_labels.csv')

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

# select only rows where mixed_gender is true

annodemogmix = annodemog[annodemog.mixed_gender]

# merge annodemog with comments

anndemcom = pd.merge(annodemogmix, comments)

# remove duplicate comments

anncom = anndemcom.drop_duplicates('rev_id')

# create new label with random males scores

print 'relabelling data with random male scores'

# make a new column of annotations comprising randomly sampled male annotations with replacement

anncom['toxicity_score'] = anncom.rev_id.apply(getRandoMales)

# make a new column saying whether toxic or not

anncom['toxicity'] = np.where(anncom['toxicity_score']<0, True, False)

# remove newlines and tabs

anncom['comment'] = anncom['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
anncom['comment'] = anncom['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

train_comments = anncom.query("split=='train'")
test_comments = anncom.query("split=='test'")

clf = Pipeline([
    ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
    ('tfidf', TfidfTransformer(norm = 'l2')),
    ('clf', LogisticRegression()),
])
clf = clf.fit(train_comments['comment'], train_comments['toxicity'])

filename = 'male_clf_%s.sav' % seed
pickle.dump(clf, open(filename, 'wb'))

# # normal test data
# print 'normal test...'
# auc = roc_auc_score(test_comments['toxicity'], clf.predict_proba(test_comments['comment'])[:, 1])

# test_pred_normal = clf.predict(test_comments['comment'])
# test_true_normal = test_comments['toxicity']

# print confusion_matrix(test_pred_normal, test_true)
# print auc

# # demographic subset test data
# test_comments_subset = test_detox

# print 'test on male'

# # test on male test data
# auc = roc_auc_score(test_comments_subset['male_toxicity'], clf.predict_proba(test_comments_subset['comment'])[:, 1])
# test_pred_male = clf.predict(test_comments_subset['comment'])
# test_true_male = test_comments_subset['male_toxicity']

# print confusion_matrix(test_pred_female, test_true_female)
# print auc

# print 'test on female'

# # test on female test data
# auc = roc_auc_score(test_comments_subset['female_toxicity'], clf.predict_proba(test_comments_subset['comment'])[:, 1])
# test_pred_female = clf.predict(test_comments_subset['comment'])
# test_true_female = test_comments_subset['female_toxicity']

# print confusion_matrix(test_pred_female, test_true_female)
# print auc

# print 'test on impermium dataset'

# auc = roc_auc_score(imper['toxicity'], clf.predict_proba(imper['comment'])[:, 1])
# test_pred_imper = clf.predict(imper['comment'])
# test_true_imper = imper['toxicity']

# print confusion_matrix(test_pred_imper, test_true_imper)
# print auc