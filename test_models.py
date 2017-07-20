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

output_file = open("test_results.csv", "a")
headers = "bot_id,male_auc,female_auc,imper_auc,normal_auc,TP_male,TP_female,TP_imper,TP_norm,FP_male,FP_female,FP_imper,FP_norm,FN_male,FN_female,FN_imper,FN_norm,TN_male,TN_female,TN_imper,TN_norm\n"
output_file.write(headers)
output_file.close()

models = glob.glob('models/mixed/*.sav')

print models

for model in models:
	# os.system("python test_model_mixed.py %s" % model)
	# # or for normal test
	os.system("python test_model.py %s" % model)


# # load up the training data and demographics of workers
# comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t')
# annodemog = pd.read_csv('annodemog_mixlabels.csv', sep = '\,')
# test_detox = pd.read_csv('test_detox.csv')
# imper = pd.read_csv('datasets/impermium_offence_labels.csv')

# pickled_model = open(sys.argv[1], 'rb')

# clf = pickle.load(pickled_model)

# test_detox = pd.read_csv('test_detox.csv')

# # # normal test data
# print 'normal test...'
# normal_auc = roc_auc_score(test_detox['toxicity'], clf.predict_proba(test_detox['comment'])[:, 1])
# test_pred_normal = clf.predict(test_detox['comment'])
# test_true_normal = test_detox['toxicity']
# normal_confusion_matrix = confusion_matrix(test_pred_normal, test_true_normal, labels=[1,0])
# print normal_auc
# print normal_confusion_matrix

# # demographic subset test data

# print 'test on male'

# # test on male test data
# male_auc = roc_auc_score(test_detox['male_toxicity'], clf.predict_proba(test_detox['comment'])[:, 1])
# test_pred_male = clf.predict(test_detox['comment'])
# test_true_male = test_detox['male_toxicity']
# male_confusion_matrix = confusion_matrix(test_pred_male, test_true_male, labels=[1,0])
# print male_auc
# print male_confusion_matrix

# print 'test on female'

# # test on female test data
# female_auc = roc_auc_score(test_detox['female_toxicity'], clf.predict_proba(test_detox['comment'])[:, 1])
# test_pred_female = clf.predict(test_detox['comment'])
# test_true_female = test_detox['female_toxicity']
# female_confusion_matrix = confusion_matrix(test_pred_female, test_true_female, labels=[1,0])
# print female_auc
# print female_confusion_matrix

# # print 'test on impermium dataset'

# imper_auc = roc_auc_score(imper['toxicity'], clf.predict_proba(imper['comment'])[:, 1])
# test_pred_imper = clf.predict(imper['comment'])
# test_true_imper = imper['toxicity']
# imper_confusion_matrix = confusion_matrix(test_pred_imper, test_true_imper, labels=[1,0])
# print imper_auc
# print imper_confusion_matrix

# print 'computin` fairness measures...'

# # ratio of M/F confusion table values

# TP_male = male_confusion_matrix[0][0]
# TP_female = female_confusion_matrix[0][0]
# TP_fair = TP_male / float(TP_female)
# FP_male = male_confusion_matrix[0][1]
# FP_female = female_confusion_matrix[0][1]
# FP_fair = FP_male / float(FP_female)
# FN_male = male_confusion_matrix[1][0] 
# FN_female = female_confusion_matrix[1][0]
# FN_fair = FN_male / float(FN_female)
# TN_male = male_confusion_matrix[1][1]
# TN_female = female_confusion_matrix[1][1]
# TN_fair = TN_male / float(TN_female)
# print 'true positive male / female: %s' % TP_fair
# print 'false positive male / female: %s ' % FP_fair
# print 'false negative male / female: %s ' % FN_fair
# print 'true negative male / female: %s' % TN_fair

# output = "bot_id,gender,seed,TP_male,TP_female,"

# """
# the cf tables are as such:

# TP, FP 
# FN, TN

# """