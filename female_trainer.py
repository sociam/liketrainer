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

# load up the training data and demographics of workers
comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t')
annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')
demog = pd.read_csv('toxicity_worker_demographics.tsv', sep = '\t')

print len(annotations['rev_id'].unique())

# merge everything, select only female workers
demog_annotations = pd.merge(annotations, demog, on='worker_id')
demog_comments_annotations = pd.merge(comments, demog_annotations, on='rev_id')
female_only = demog_comments_annotations[demog_comments_annotations.gender == 'female']

# remove unwanted columns

for column in ['year', 'logged_in', 'ns', 'sample', 'toxicity', 'english_first_language', 'education']:
	del female_only[column]

# create a new column with toxicity score, i.e. the mean average toxicity rating for that comment

female_only['toxicity_average'] = female_only['toxicity_score'].groupby(female_only['rev_id']).transform(np.mean)

# create a new column with a toxicity classification, which is true if the toxixity score < 0

female_only['toxicity_class'] = female_only['toxicity_average'] < 0.00

# remove newlines and tabs

female_only['comment'] = female_only['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
female_only['comment'] = female_only['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

print female_only

train_female_only = female_only.query("split=='train'")
test_female_only = female_only.query("split=='test'")

clf = Pipeline([
    ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
    ('tfidf', TfidfTransformer(norm = 'l2')),
    ('clf', LogisticRegression()),
])
clf = clf.fit(train_female_only['comment'], train_female_only['toxicity_class'])

auc = roc_auc_score(test_female_only['toxicity_class'], clf.predict_proba(test_female_only['comment'])[:, 1])

prediction1 = clf.predict(['you suck, go away you pathetic loser, I hate you!'])
prediction2 = clf.predict(['happy world. nice words. very great.'])
print prediction1
print prediction2

# for ideological books corpus

num_benign_bipartisan_neutral = 0
num_toxic_bipartisan_neutral = 0

[lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
for tree in neutral[0:600]:
	treewords = tree.get_words()
	print treewords
	toxicity = clf.predict([treewords])
	print toxicity
	if str(toxicity) == '[False]':
		num_toxic_bipartisan_neutral = num_toxic_bipartisan_neutral +1
		print 'added to toxic'
	if str(toxicity) == '[ True]':
		num_benign_bipartisan_neutral = num_benign_bipartisan_neutral +1
		print 'added to benign'

num_benign_liberal = 0
num_toxic_liberal = 0

[lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
for tree in lib[0:1700]:
	treewords = tree.get_words()
	print treewords
	toxicity = clf.predict([treewords])
	print toxicity
	if str(toxicity) == '[False]':
		num_toxic_liberal = num_toxic_liberal +1
		print 'added to toxic'
	if str(toxicity) == '[ True]':
		num_benign_liberal = num_benign_liberal +1
		print 'added to benign'


num_benign_con = 0
num_toxic_con = 0

[lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
for tree in con[0:1700]:
	treewords = tree.get_words()
	print treewords
	toxicity = clf.predict([treewords])
	print toxicity
	if str(toxicity) == '[False]':
		num_toxic_con = num_toxic_con +1
		print 'added to toxic'
	if str(toxicity) == '[ True]':
		num_benign_con = num_benign_con +1
		print 'added to benign'


print 'neutral:'

print num_benign_bipartisan_neutral
print num_toxic_bipartisan_neutral

print 'con:'

print num_benign_con
print num_toxic_con

print 'liberal:'

print num_benign_liberal
print num_toxic_liberal

print 'tests'
print prediction1
print prediction2

print('Test ROC AUC: %.3f' %auc)

# get average toxiticity score for bipartisan convote data

# num_benign_bipartisan = 0
# num_toxic_bipartisan = 0

# for filename in glob.iglob('training_set/*.txt'):
# 	speech = open(filename, 'r')
# 	speechtext = speech.read()
# 	toxicity = str(clf.predict([speechtext]))
# 	print toxicity
# 	if toxicity == '[False]':
# 		num_benign_bipartisan = num_benign_bipartisan + 1
# 		print 'added to benign'
# 	if toxicity == '[ True]':
# 		num_toxic_bipartisan = num_toxic_bipartisan + 1
# 		print 'added to toxic'

# print num_benign_bipartisan
# print num_toxic_bipartisan