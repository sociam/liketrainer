


# requires nltk and the corpus dataset
# run nltk.download() to download the dataset before running this

import nltk, re
from nltk.corpus import nps_chat
from collections import Counter
import make_clf

DEBUG_CLASSIFICATIONS = True

## if your nltk_data is not in the normal place ... 
def addpath(p):
	nltk.data.path.append(p)

not_channel_op = lambda m : not ' '.join(m) in ['PART', 'JOIN']

def process_fid(fid, by_demo):
	matches = re.search('(\d+)-(\d+)-(.*)_(.*).xml', fid)
	(mon, day, demo, posts) = (matches.group(1), matches.group(2), matches.group(3), matches.group(4))
	# print 'month ', mon, ' day ', day, ' demo ', demo, ' posts ', posts
	by_demo[demo] = (by_demo.get(demo) or []) + filter(not_channel_op, nps_chat.posts(fid)[:])


# loads the corpus into a dict by demographic { demo1: [msg1, msg2] }
def load_by_demo():
	by_demo = {}
	[process_fid(fid, by_demo) for fid in nps_chat.fileids()]
	return by_demo

# add more paths here, won't hurt if the path doesn't exist on your computer
# (nltk is a bad boy and likes installing in shit places)
addpath('/Users/electronic/Desktop/liketrainer/nltk_data')

corpus_by_demographic = load_by_demo()
# print make_clf.clf.predict(['happy world. nice words. very great.'])

joinup = lambda p: ' '.join(p)
predict = lambda s: make_clf.clf.predict([s])[0]

# iterates through all demographics in the corpus and applies classifier true/false
# for demographic, msgs in corpus_by_demographic.items():
# 	freqs[demographic] = dict(Counter([make_clf.clf.predict([' '.join(x)])[0] for x in msgs]))

freqs = {}
for demographic, msgs in corpus_by_demographic.items():
	if (DEBUG_CLASSIFICATIONS):
		for m in msgs:
			print '%s : %s ' % (joinup(m), str(predict(joinup(m))))

	freqs[demographic] = dict(Counter([predict(joinup(x)) for x in msgs]))
print freqs
