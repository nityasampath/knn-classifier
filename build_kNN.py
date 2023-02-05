#!/usr/bin/env python3

import sys
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances
from numpy import argsort

#get files and values from passed arguments
training_file = sys.argv[1]
test_file = sys.argv[2]
k_val = int(sys.argv[3])
sim = int(sys.argv[4])
if sim == 1: #Euclidean distance
	similarity_func = 'euclidean'
elif sim == 2: #cosine distance
	similarity_func = 'cosine'

sys_output = open(sys.argv[5], 'w')

training_insts = [] #list of dictionaries for each instance, {word : count}
training_labels = [] #list of labels, label at each index corresponds to training instance at same index
test_insts = [] #list of dictionaries for each instance, {word : count}
test_labels = [] #list of labels, label at each index corresponds to test instance at same index

#training stage
#read in training data
with open(training_file, 'r') as train_data:

	linecount = 0

	for line in train_data:
		line = line.strip()
		tokens = line.split(' ')
		
		label = tokens[0]

		inst_id = 'inst' + str(linecount)
		linecount += 1

		inst_vec = {}
		inst_label = label

		for token in tokens[1:]:
			pair = token.split(':')
			word = pair[0]
			val = int(pair[1])
			inst_vec[word] = val

		training_insts.append(inst_vec)
		training_labels.append(inst_label)


#read in test data
with open(test_file, 'r') as test_data:

	linecount = 0

	for line in test_data:
		line = line.strip()
		tokens = line.split(' ')
		
		label = tokens[0]

		inst_vec = {}
		inst_label = label

		for token in tokens[1:]:
			pair = token.split(':')
			word = pair[0]
			val = int(pair[1])
			inst_vec[word] = val

		test_insts.append(inst_vec)
		test_labels.append(inst_label)

#vectorize test instance & training instances
V = DictVectorizer(sparse=True)
training_matrix = V.fit_transform(training_insts)
test_matrix = V.transform(test_insts)

#test stage
def classify(training_mat, test_mat, data_label):
	sys_output.write('%%%%% ' + data_label + ':' + '\n')

	if data_label == 'training data':
		labels_list = training_labels
	if data_label == 'test data':
		labels_list = test_labels

	#to store correct and incorrect counts to calculate accuracy
	confusion_matrix = [[0 for i in range(3)] for j in range(3)]
	i = 0
	j = 0

	#calculate distances between the test instances and the training instances
	distances = pairwise_distances(test_mat, Y=training_mat, metric=similarity_func)

	#sort distances
	sorted_distances = argsort(distances)

	for i in range(len(sorted_distances)): #iterate through each document
		k_neighbors = []

		for j in range(k_val): #get k nearest neighbors
			k_neighbors.append(training_labels[sorted_distances[i][j]])

		label_counts = {'talk.politics.guns' : 0, 'talk.politics.mideast' : 0, 'talk.politics.misc' : 0}
		for neighbor_label in k_neighbors:
			label_counts[neighbor_label] += 1

		#calculate probabilites for each class
		probs = {}
		labels_sum = sum(label_counts.values())
		for label in label_counts:
			probs[label] = label_counts[label]/labels_sum

		system_out = max(probs, key=lambda key: probs[key])

		true_label = labels_list[i]

		#count incorrect and correct responses
		if true_label == 'talk.politics.guns':
			tl = 0
		elif true_label == 'talk.politics.misc':
			tl = 1
		elif true_label == 'talk.politics.mideast':
			tl = 2

		if system_out == 'talk.politics.guns':
			so = 0
		elif system_out == 'talk.politics.misc':
			so = 1
		elif system_out == 'talk.politics.mideast':
			so = 2

		confusion_matrix[tl][so] = confusion_matrix[tl][so] + 1

		#sort probabilities dictionary by value to print in order
		sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)

		#print to sys_output file
		output = ''
		for item in sorted_probs:
			label = item[0]
			prob = item[1]
			output += ' ' + label + ' ' + str(prob)

		sys_output.write('array:' + str(i) + ' ' + true_label + str(output) + '\n')

	sys_output.write('\n')

	#calculate accuracy from confusion matrix values
	total = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2]
	correct = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] 
	accuracy = correct/total

	print('Confusion matrix for the ' + data_label + ':\nrow is the truth, column is the system output')
	print()
	print('             talk.politics.guns talk.politics.misc talk.politics.mideast')
	print('talk.politics.guns ' + str(confusion_matrix[0][0]) + ' ' + str(confusion_matrix[0][1]) + ' ' + str(confusion_matrix[0][2]))
	print('talk.politics.misc ' + str(confusion_matrix[1][0]) + ' '  + str(confusion_matrix[1][1]) + ' '  + str(confusion_matrix[1][2]))
	print('talk.politics.mideast ' + str(confusion_matrix[2][0]) + ' '  + str(confusion_matrix[2][1]) + ' '  + str(confusion_matrix[2][2]))
	print()
	if data_label == 'training data':
		print(' Training accuracy= ' + str(accuracy))
	if data_label == 'test data':
		print(' Test accuracy= ' + str(accuracy))
	print()
	print()

classify(training_matrix, training_matrix, 'training data')
classify(training_matrix, test_matrix, 'test data')

sys_output.close()
