import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os
import glob
import pandas as pd
import sys  
import cPickle

print "making clf .... "
comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')

print len(annotations['rev_id'].unique())

# labels a comment as toxic if the majority of annoatators gave it below 0 (neutral) toxicity score
labels = annotations.groupby('rev_id')['toxicity_score'].mean() < 0.0

# join labels and comments
comments['toxicity'] = labels

comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

train_comments = comments.query("split=='train'")
test_comments = comments.query("split=='test'")

clf = Pipeline([
    ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
    ('tfidf', TfidfTransformer(norm = 'l2')),
    ('clf', LogisticRegression()),
])
clf = clf.fit(train_comments['comment'], train_comments['toxicity'])

auc = roc_auc_score(test_comments['toxicity'], clf.predict_proba(test_comments['comment'])[:, 1])

print "done .... "
prediction1 = clf.predict(['you suck, go away you pathetic loser, I hate you!'])
prediction2 = clf.predict(['happy world. nice words. very great.'])

