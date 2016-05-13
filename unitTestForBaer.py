#!/usr/bin/env python

import sys
import baer

if not len(sys.argv) > 1:
	print __doc__
	sys.exit(1)

try:
	times = int(sys.argv[1])
except ValueError:
	print "Not correct argument is entered"
	print __doc__
	sys.exit(1)

filename = 'pima-indians-diabetes.data.csv'
splitRatio = 0.67
dataset = baer.loadCsv(filename)
acc = []

for i in range(times):
	# baer.main()

	trainingSet, testSet = baer.splitDataset(dataset, splitRatio)
	summaries = baer.summarizeByClass(trainingSet)
	predictions = baer.getPredictions(summaries, testSet)
	accuracy = baer.getAccuracy(testSet, predictions)
	
	acc.append(accuracy)
	print('\nAccuracy: {0} %').format(accuracy)

summary = 0.0
for i in acc:
	summary += i

result = summary / len(acc)
print "\nThe average is: %s" % result