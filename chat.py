
# requires nltk and the corpus dataset
# run nltk.download() to download the dataset before running this

import nltk, re
from nltk.corpus import nps_chat

## if your nltk_data is not in the normal place ... 
def addpath(p):
	nltk.data.path.append(p)


def process_fid(fid, by_demo):
	matches = re.search('(\d+)-(\d+)-(.*)_(.*).xml', fid)
	(mon, day, demo, posts) = (matches.group(1), matches.group(2), matches.group(3), matches.group(4))
	# print 'month ', mon, ' day ', day, ' demo ', demo, ' posts ', posts
	by_demo[demo] = (by_demo.get(demo) or []) + nps_chat.posts(fid)[:]

def load_by_demo():
	by_demo = {}
	[process_fid(fid, by_demo) for fid in nps_chat.fileids()]
	return by_demo

# add more paths here, won't hurt if the path doesn't exist on your computer
addpath('/Users/electronic/Desktop/liketrainer/nltk_data')

print load_by_demo()

# prints a dict where the key is the demographic, values are the posts

