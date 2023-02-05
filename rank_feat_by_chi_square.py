#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.feature_extraction import DictVectorizer

features = {} #{word : {class : doc count}}
classes = {} #{class : doc count}

feat_chiScore = {}

#read in data
for line in sys.stdin:
	line = line.strip()
	tokens = line.split(' ')
	
	label = tokens[0]
	if label in classes:
		classes[label] += 1 
	else:
		classes[label] = 1

	for token in tokens[1:]:
		pair = token.split(':')
		word = pair[0]
		if word in features:
			if label in features[word]:
				features[word][label] += 1
			else:
				features[word][label] = 1
		else:
			features[word] = {label : 1}

#iterate through features to calculate chi-squared
for word in features:
	#observed
	observed_feat_counts = features[word]
	observed_not_counts = {}
	for label, count in observed_feat_counts.items():
		observed_not_counts[label] = classes[label] - features[word][label]
	observed = [observed_not_counts, observed_feat_counts]
	
	V = DictVectorizer(sparse=False)
	observed_matrix = V.fit_transform(observed)

	#expected
	totals_row = np.sum(observed_matrix, axis=0, keepdims=True)
	totals_column = np.sum(observed_matrix, axis=1, keepdims=True)
	totals = np.sum(observed_matrix)

	expected_matrix = (totals_row * totals_column)/totals

	#calculate chi-squared
	chi_squared_matrix = ((observed_matrix - expected_matrix)**2)/expected_matrix
	chi_squared = np.sum(chi_squared_matrix)

	feat_chiScore[word] = chi_squared

#sort by highest chi-squares
sorted_feats = sorted(feat_chiScore.items(), key=lambda item: item[1], reverse=True)

#print features
for feat in sorted_feats:
	word = feat[0]
	chi_squared = feat[1]
	docNum = sum(features[word].values())
	print(word + ' ' + str(chi_squared) + ' ' + str(docNum))








