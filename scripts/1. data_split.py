### This script does the following:
### (1) Check the number of duplicates in original dataset; resplit all data if necessary
### (2) If no duplicates, or the number of duplicates is small, keep the original test set from the relevant papers, 
###     then split the residual data with different split methods
###     a. random
###     b. heuristic
###     c. adversarial
### (3) generate test set cross-validation style; 5 test set, 3 test/dev splits per test set with set sample sizes


import io, argparse, os, random, statistics
from collections import Counter
from scipy.stats import wasserstein_distance

import collections
import random
from typing import Dict, Generator, Iterator, List, Set, Text, Tuple

#from absl import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_extraction
from sklearn import neighbors

random.seed(8)

lgs = ['mayo', 'mexicanero', 'nahuatl', 'wixarika', 'shp', 'tar', 'hupa', 'popoluca', 'tepehua', 'aka', 'swa', 'tam', 'tgl', 'seneca', 'english', 'german', 'finnish', 'indonesian', 'turkish']
lg_maps = {'mayo': 'Yorem Nokki', 'mexicanero': 'Mexicanero', 'nahuatl': 'Nahuatl', 'wixarika': 'Wixarika', 'shp': 'Shipibo-Konibo', 'tar': 'Raramuri', 'hupa': 'Hupa', 'aka': 'Akan', 'swa': 'Swahili', 'tam': 'Tamil', 'tgl': 'Tagalog'}
#lg_resource_dir = {'indlangs': ['mayo', 'mexicanero', 'nahuatl', 'wixarika']: 'morpho-data': ['shp', 'tar']}


if not os.path.exists('configs/'):
	os.system('mkdir configs/')

### Check the number of duplicated words in the full dataset of each language ###

def check_duplicates(lg):
	data = []
	test_data = []
	for file in os.listdir('data/' + lg + '/original/'):
		if os.path.isfile('data/' + lg + '/original/' + file) and file.endswith('src'):
			with open('data/' + lg + '/original/' + file) as f:
				for line in f:
					toks = line.strip()
					data.append(toks)
			if 'test' in file:
				with open('data/' + lg + '/original/' + file) as f:
					for line in f:
						toks = line.strip()
						test_data.append(toks)

	print(len(data))
	print(len(test_data))

	unique_words = set(data)
	if len(unique_words) != len(data):
		print(lg, len(unique_words), len(data))

	unique_test_words = set(test_data)
	if len(unique_test_words) != len(test_data):
		print(lg, len(unique_test_words), len(test_data))


### Gather all source and target data from each language, after removal of duplicates
### Justification: in language documentation, we can just assign the gold-standard segmentation to any seen words

def gather_data(lg):
	source_target_data = {}
	source_data = []
	target_data = []
	for file in os.listdir('data/' + lg + '/original/'):
		if os.path.isfile('data/' + lg + '/original/' + file):
			if lg not in ['popoluca', 'tepehua', 'hupa', 'seneca', 'aka', 'swa', 'tam', 'tgl']:
				if file.endswith('src'):

					filename = file[ : -4]									
					with open('data/' + lg + '/original/' + file) as f:
						for line in f:
							toks = line.strip()
							if lg not in ['shp', 'tar']:
								source_data.append(toks)
							else:
								source_data.append(' '.join(c for c in toks))

					if lg not in ['shp', 'tar']:
						with open('data/' + lg + '/original/' + filename + '_trg') as f:
							for line in f:
								toks = line.strip()
								target_data.append(toks)
					else:
						with open('data/' + lg + '/original/' + filename + '.tgt') as f:
							for line in f:
								toks = line.strip()
								toks = toks.split()
								toks = '!'.join(seg for seg in toks)
								target_data.append(' '.join(c for c in toks))

			elif lg in ['popoluca', 'tepehua']:
				with open('data/' + lg + '/original/' + file, encoding = 'latin-1') as f:
					for line in f:
						toks = line.strip().split()
						source_data.append(' '.join(c for c in toks[0]))
						tgt_w = toks[1].split('-')
						tgt_w = '!'.join(morph for morph in tgt_w)
						target_data.append(' '.join(c for c in tgt_w))

			else:
				if file.endswith('src'):
					with open('data/' + lg + '/original/' + file) as f:
						for line in f:
							toks = line.strip()
							source_data.append(toks)
				if file.endswith('tgt'):
					with open('data/' + lg + '/original/' + file) as f:
						for line in f:
							toks = line.strip()
							target_data.append(toks)

	assert len(source_data) == len(target_data)

	for i in range(len(source_data)):
		source_w = source_data[i]
		target_w = target_data[i]

		### Excluding duplicated words 

		if source_w not in source_target_data:
			source_target_data[source_w] = target_w

	target_vocab = ['<blank>', '<s>', '</s>']
	for w in target_data:
		w = w.split()
		for c in w:
			if c not in target_vocab:
				target_vocab.append(c)

	if not os.path.exists('data/' + lg + '/vocab/'):
		os.system('mkdir data/' + lg + '/vocab/')

	with open('data/' + lg + '/vocab/' + lg + '_src_vocab', 'w', encoding = 'utf-8') as f:
		for c in target_vocab:
			if c != '!':
				f.write(c + '\n')

	with open('data/' + lg + '/vocab/' + lg + '_tgt_vocab', 'w', encoding = 'utf-8') as f:
		for c in target_vocab:
			f.write(c + '\n')
	print(len(source_target_data))
	return source_target_data


### Think about to-do: cross-validation of residual data ###

### Random split ###

def random_split(residual_data, n=0, dev_proportion = 0.1):
	idx_list = []

	split_size = n
 
 	# if sample size is set the spilt is equal to sample size plus dev size so once dev is removed, approriate sample size remains
	# if n == 0 (False) then ignore and use residual data len
	if not split_size: split_size = len(residual_data)
	dev_size = round(n * dev_proportion)

	for i in range(split_size):
		idx_list.append(i)

	random.shuffle(idx_list)

	dev_idx_list = random.sample(idx_list, dev_size)
	dev_data = []
	for idx in dev_idx_list:
		dev_data.append(residual_data[idx])
	train_data = [tok for tok in residual_data if tok not in dev_data]

	return train_data, dev_data


### Splitting by threshold of number of morphemes ###

def split_by_morph_threshold(residual_data, dev_proportion = 0.1):

	residual_target_data = [tok[1] for tok in residual_data]
	num_morph_list = [w.count('!') + 1 for w in residual_target_data]
	dev_size = round(len(residual_data) * dev_proportion)

	current_count = 0
	check = 'no'
	final_threshold = ''

	# Start from the longest texts.
	for threshold in range(max(num_morph_list), 0, -1):
		current_count += num_morph_list.count(threshold)
		if current_count > dev_size:
			ratio = current_count / len(num_morph_list)
			if ratio >= dev_proportion - 0.02 and ratio <= dev_proportion + 0.02:
				check = 'yes'
				final_threshold = threshold
	
	if check == 'yes':
		dev_data = [tok for tok in residual_data if tok[1].count('!') + 1 >= threshold]
		train_data = [tok for tok in residual_data if tok not in dev_data]

		return train_data, dev_data

	return None


### Splitting by threshold of average morpheme length ###

def split_by_length_threshold(residual_data, dev_proportion = 0.1):

	residual_target_data = [tok[1] for tok in residual_data]
	ave_morph_len_list = []
	for w in residual_data:
		w = ''.join(c for c in w.split()).split('!')
		morph_len = []
		for morph in  w:
			morph_len.append(len(morph))
		ave_morph_len = sum(morph_len) / len(morph_len)
		ave_morph_len_list.append(ave_morph_len)

	dev_size = round(len(residual_data) * dev_proportion)

	current_count = 0
	check = 'no'
	final_threshold = ''

	# Start from the longest texts.
	for threshold in range(max(ave_morph_len_list), 0, -1):
		current_count += ave_morph_len_list.count(threshold)
		if current_count > dev_size:
			ratio = current_count / len(ave_morph_len_list)
			if ratio >= dev_proportion - 0.02 and ratio <= dev_proportion + 0.02:
				check = 'yes'
				final_threshold = threshold
	
	if check == 'yes':
		dev_data = [tok for tok in residual_data if tok[1].count('!') + 1 >= threshold]
		train_data = [tok for tok in residual_data if tok not in dev_data]

		return train_data, dev_data

	return None

### Adversarial splitting ###

"""
Finds test sets by maximizing Wasserstein distances among the given texts.
This is separating the given texts into training/dev and test sets based on an
approximate Wasserstein method. First all texts are indexed in a nearest
neighbors structure. Then a new test centroid is sampled randomly, from which
the nearest neighbors in Wasserstein space are extracted. Those constitute
the new test set.
Similarity is computed based on document-term counts.
Args:
	texts: Texts to split into training/dev and test sets.
	test_set_size: Number of elements the new test set should contain.
	no_of_trials: Number of test sets requested.
	min_df: Mainly for speed-up and memory efficiency. All tokens must occur at
	  least this many times to be considered in the Wasserstein computation.
	leaf_size: Leaf size parameter of the nearest neighbor search. Set high
	  values for slower, but less memory-heavy computation.
Returns:
	Returns a List of test set indices, one for each trial. The indices
	correspond to the items in `texts` that should be part of the test set.
"""
			
def split_with_wasserstein(residual_data, dev_proportion = 0.1, min_df = 1, leaf_size = 3):

	residual_target_data = [' '.join(morph for morph in ''.join(c for c in tok[1].split()).split('!')) for tok in residual_data]
	dev_size = round(len(residual_target_data) * dev_proportion)

	vectorizer = feature_extraction.text.CountVectorizer(dtype=np.int8, min_df=min_df)
#	logging.info('Creating count vectors.')
	text_counts = vectorizer.fit_transform(residual_target_data)
	text_counts = text_counts.todense()
#	logging.info('Count vector shape %s.', text_counts.shape)
#	logging.info('Creating tree structure.')
	nn_tree = neighbors.NearestNeighbors(n_neighbors = dev_size, algorithm = 'ball_tree', leaf_size = leaf_size, metric = stats.wasserstein_distance)
	nn_tree.fit(np.asarray(text_counts))
#	logging.info('Sampling test sets.')

	# Sample random test centroid.
	sampled_poind = np.random.randint(text_counts.max().max() + 1, size = (1, text_counts.shape[1]))
	nearest_neighbors = nn_tree.kneighbors(sampled_poind, return_distance = False)
	# I queried for only one datapoint.
	nearest_neighbors = nearest_neighbors[0]
#	logging.info(nearest_neighbors[:10])
	dev_indices = list(nearest_neighbors)
	dev_data = [residual_data[idx] for idx in dev_indices]
	train_data = [tok for tok in residual_data if tok not in dev_data]

	return train_data, dev_data


### Write files

def write_output(lg, train_data, dev_data, test_proportion, test_idx, idx, split, n):
	sample = str(n)

	with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.src', 'w', encoding = 'utf-8') as f:
		source_train = [tok[0] for tok in train_data]
		for w in source_train:
			f.write(w + '\n')

	with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.tgt', 'w', encoding = 'utf-8') as f:
		target_train = [tok[1] for tok in train_data]
		for w in target_train:
			f.write(w + '\n')

	with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.src', 'w', encoding = 'utf-8') as f:
		source_dev = [tok[0] for tok in dev_data]
		for w in source_dev:
			f.write(w + '\n')

	with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.tgt', 'w', encoding = 'utf-8') as f:
		target_dev = [tok[1] for tok in dev_data]
		for w in target_dev:
			f.write(w + '\n')

	if not os.path.exists('configs/' + lg):
		os.system('mkdir configs/' + lg)

	if not os.path.exists('configs/' + lg + '/' + method + '_' + sample):
		os.system('mkdir configs/' + lg + '/' + method + '_' + sample)

	if not os.path.exists('configs/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100))):
		os.system('mkdir configs/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)))

	with open('configs/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.yaml', 'w', encoding = 'utf-8') as f:
		f.write('save_data: data' + '\n')
		f.write('src_vocab: data/' + lg + '/vocab/' + lg + '_src_vocab' + '\n')
		f.write('tgt_vocab: data/' + lg + '/vocab/' + lg + '_tgt_vocab' + '\n')
		f.write('overwrite: False' + '\n')
		f.write('\n')
		f.write('data:' + '\n')
		f.write('  ' + 'train:' + '\n')
		f.write('  ' + '  ' + 'path_src: data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.src' + '\n')
		f.write('  ' + '  ' + 'path_tgt: data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.tgt' + '\n')
		f.write('  ' + 'valid:' + '\n')
		f.write('  ' + '  ' + 'path_src: data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.src' + '\n')
		f.write('  ' + '  ' + 'path_tgt: data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(test_idx + 1) + '_' + split + '_' + str(idx + 1) + '.tgt' + '\n')
		f.write('\n')

### Generate training/dev/test data ###

def generate_data(lg, data, test_proportion, method,  n = 0, dev_proportion = 0.1):

	data = [[k, v] for k, v in data.items()]
	random.shuffle(data)

	n_folds = round(1 / test_proportion)
	sample = str(n)
	n_words_per_fold = round(len(data) * test_proportion)

#	for i in range(n_folds):
#		start_idx = n_words_per_fold * i
#		end_idx = n_words_per_fold * (i + 1)
#		fold_data = data[start_idx: end_idx]
#		residual_data = [tok for tok in data if tok not in fold_data]

#		if not os.path.exists('data/' + lg + '/test_' + str(int(test_proportion * 100))):
#			os.system('mkdir data/' + lg + '/test_' + str(int(test_proportion * 100)))
	
#		with open('data/' + lg + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.src', 'w', encoding = 'utf-8') as f:
#			source_test = [tok[0] for tok in fold_data]
#			for w in source_test:
#				f.write(w + '\n')

#		with open('data/' + lg + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.tgt', 'w', encoding = 'utf-8') as f:
#			target_test = [tok[1] for tok in fold_data]
#			for w in target_test:
#				f.write(w + '\n')

		### Generate training and dev data, random + adversarial splits 
#		for n_split in range(3):
#			train_data, dev_data = random_split(residual_data, dev_proportion)
#			write_output(lg, train_data, dev_data, test_proportion, i, n_split, 'random')

#			train_data, dev_data = split_with_wasserstein(residual_data, dev_proportion)
#			write_output(lg, train_data, dev_data, test_proportion, i, n_split, 'adversarial')

		### Generate training and dev data, heuristic splits

#		try:
#			train_data, dev_data = split_by_morph_threshold(residual_data, dev_proportion)
#			write_output(lg, train_data, dev_data, test_proportion, i, 0, 'morph')
#		except:
#			print('Split by number of morphemes was not successful')

#		try:
#			train_data, dev_data = split_by_length_threshold(residual_data, dev_proportion)
#			write_output(lg, train_data, dev_data, test_proportion, i, 0, 'length')
#		except:
#			print('Split by average morpheme lengths was not successful')

#	if n_folds != 10:
	morph_c = 0
	length_c = 0
	folds = []
	if 2 > 1:
		data_pool = data
		for i in range(n_folds): ### for each fold one test split is made and remaining data is used as the sources for 3 random training samples
			fold_data = ''

			## Randomly generating new test sets
			if method == 'random':
				fold_data = random.sample(data_pool, n_words_per_fold) 
				
			## Adversarially generating new test sets
			else:
				_, fold_data = split_with_wasserstein(data_pool, test_proportion)

			folds.append(fold_data)
			data_pool = [tok for tok in data_pool if tok not in fold_data]

		for i in range(n_folds): ### for each fold one test split is made and remaining data is used as the sources for 3 random training samples
			fold_data = folds[i]
			training_pool = ''

			training_pool = [tok for tok in data if tok not in fold_data]

			if not os.path.exists('data/' + lg + '/' + method + '_' + sample):
				os.system('mkdir data/' + lg + '/' + method + '_' + sample)

			if not os.path.exists('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100))):
				os.system('mkdir data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)))
	
			with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.src', 'w', encoding = 'utf-8') as f:
			#	if method == 'random':
				source_test = [tok[0] for tok in fold_data]
			#	for tok in source_test:
			#		print(tok)
				for w in source_test:
					f.write(w + '\n')
			#	else:
			#		for tok in fold_data:

			with open('data/' + lg + '/' + method + '_' + sample + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.tgt', 'w', encoding = 'utf-8') as f:
				target_test = [tok[1] for tok in fold_data]
				for w in target_test:
					f.write(w + '\n')

			### Generate training and dev data, random + adversarial splits 
			for n_split in range(3):
				# the n here will be the train sample size - 100, 200, etc.
				train_data, dev_data = random_split(training_pool, n, dev_proportion)
				write_output(lg, train_data, dev_data, test_proportion, i, n_split, 'random', n)

				train_data, dev_data = split_with_wasserstein(training_pool, dev_proportion)
				write_output(lg, train_data, dev_data, test_proportion, i, n_split, 'adversarial', n)

			### Generate training and dev data, heuristic splits

			try:
				train_data, dev_data = split_by_morph_threshold(training_pool, dev_proportion)
				write_output(lg, train_data, dev_data, test_proportion, i, 0, 'morph')
				morph_c += 1
			except:
				print('Split by number of morphemes was not successful')

			try:
				train_data, dev_data = split_by_length_threshold(training_pool, dev_proportion)
				write_output(lg, train_data, dev_data, test_proportion, i, 0, 'length')
				length_c += 1
			except:
				print('Split by average morpheme lengths was not successful')

	print(morph_c, length_c)

parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'language')
parser.add_argument('--test', type = str, default = '0.4', help = 'test set proportion')
parser.add_argument('--method', type = str, default = 'random', help = 'random or adversarial split to generate new test samples')
parser.add_argument('--n', type = str, default = '0', help = 'sample size for training splits')

args = parser.parse_args()

source_target_data = gather_data(args.lg)
method = args.method

generate_data(args.lg, source_target_data, float(args.test), method, int(args.n))




