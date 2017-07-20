import os

seeds = [2,3,4,5,6,7,8,9,10]

for seed in seeds:
	print 'making mixed classifer with seed %s' % seed
	os.system("python make_mixed_clf.py %s" % seed)
