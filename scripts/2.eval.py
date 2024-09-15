import statistics, io, argparse, random, os
import numpy as np
from collections import Counter


def accuracy(gold_word, pred_word):

	correct = 0

	if gold_word == pred_word:
		correct = 1

	return correct * 100 


def F1(gold_word, pred_word):

	correct_total = 0

	for m in pred_word:
		if m in gold_word:
			correct_total += 1

	gold_total = len(gold_word)
	pred_total = len(pred_word)

	precision = correct_total / pred_total
	recall = correct_total / gold_total

	F1 = 0

	try:
		F1 = 2 * (precision * recall) / (precision + recall)
		F1 = round(F1 * 100, 2)
	except:
		F1 = 0

	return round(precision * 100, 2), round(recall * 100, 2), F1

def call_counter(func):
	def helper(*args, **kwargs):
		helper.calls += 1
		return func(*args, **kwargs)
	helper.calls = 0
	helper.__name__= func.__name__

	return helper

def memoize(func):
	mem = {}
	def memoizer(*args, **kwargs):
		key = str(args) + str(kwargs)
		if key not in mem:
			mem[key] = func(*args, **kwargs)
		return mem[key]
	return memoizer

@call_counter
@memoize    
def levenshtein(s, t):
	if s == "":
		return len(t)
	if t == "":
		return len(s)
	if s[-1] == t[-1]:
		cost = 0
	else:
		cost = 1
	
	res = min([levenshtein(s[:-1], t)+1,
			   levenshtein(s, t[:-1])+1, 
			   levenshtein(s[:-1], t[:-1]) + cost])

	return res

def copy(gold_word, pred_word):

	gold_word = ''.join(m for m in gold_word)
	pred_word = ''.join(m for m in pred_word)

	if len(gold_word) == 0:
		print(pred_word)

	correct = 0

	if len(gold_word) <= len(pred_word):

		for i in range(len(gold_word)):
			if gold_word[i] == pred_word[i]:
				correct += 1

	if len(gold_word) > len(pred_word):

		for i in range(len(pred_word)):
			if gold_word[i] == pred_word[i]:
				correct += 1

	return round(correct * 100 / len(gold_word), 2)


if not os.path.exists('results/'):
	os.system('mkdir results/')

parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'language')
parser.add_argument('--test', type = str, default = '0.4', help = 'test set proportion')
parser.add_argument('--model', type = str, help = 'model type')
parser.add_argument('--n', type = str, default = '0', help = 'sample size for training splits')


args = parser.parse_args()

lg = args.lg
test_proportion = float(args.test)
model = args.model
n = args.n
n_folds = 5 #round(1 / test_proportion)

if not os.path.exists('results/' + lg + '_' + n):
	os.system('mkdir results/' + lg + '_' + n)
if not os.path.exists('results/' + lg + '_' + n + '/random'):
	os.system('mkdir results/' + lg + '_' + n + '/random')
if not os.path.exists('results/' + lg + '_' + n + '/adversarial'):
	os.system('mkdir results/' + lg + '_' + n + '/adversarial')
if not os.path.exists('results/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir results/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('results/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir results/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('results/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir results/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model)
if not os.path.exists('results/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir results/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model)

def eval(gold_file, pred_file):
	seed_accuracy = []
	seed_precision = []
	seed_recall = []
	seed_f1 = []
	seed_dist = []
	seed_copy = []

	
	gold_data = []
	with open(gold_file) as f:
		for line in f:
			toks = line.strip().split()
			toks = ''.join(c for c in toks)
			morphs = toks.split('!')
			gold_data.append(morphs)

	pred_data = []
	with open(pred_file) as f:
		for line in f:
			toks = line.strip().split()
			toks = ''.join(c for c in toks)
			morphs = ''
			if model not in ['gru']:
				morphs = toks.split('!')
			else:
				morphs = toks.split('<unk>')
			pred_data.append(morphs)

	try:
		for i in range(len(gold_data)):
			seed_accuracy.append(accuracy(gold_data[i], pred_data[i]))
			precision, recall, f1 = F1(gold_data[i], pred_data[i])
			dist = levenshtein(' '.join(m for m in gold_data[i]), ' '.join(m for m in pred_data[i]))

			seed_precision.append(precision)
			seed_recall.append(recall)
			seed_f1.append(f1)
			seed_dist.append(dist)
			seed_copy.append(copy(gold_data[i], pred_data[i]))

		return statistics.mean(seed_accuracy), statistics.mean(seed_precision), statistics.mean(seed_recall), statistics.mean(seed_f1), statistics.mean(seed_dist), statistics.mean(seed_copy)
	
	except:
		print(gold_file, pred_file, len(gold_data), len(pred_data))
		try:
			os.system('rm ' + pred_file)
		except:
			print(pred_file)
		return None, None, None, None, None, None


#for method in ['random', 'adversarial']:
for method in ['random']:
	for split in ['random', 'adversarial']:#, 'length', 'morph']:
		for i in range(n_folds):
			for split_n in ['1', '2', '3']:
				all_dev_accuracy = []
				all_dev_precision = []
				all_dev_recall = []
				all_dev_f1 = []
				all_dev_dist = []
				all_dev_copy = []

				all_test_accuracy = []
				all_test_precision = []
				all_test_recall = []
				all_test_f1 = []
				all_test_dist = []
				all_test_copy = []

				dev_gold_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.tgt'
				dev_filename = dev_gold_file.split('.')[0]		

				for seed in ['1', '2', '3']:

					### Evaluating predictions for dev file
			
					dev_pred_file = 'preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_filename.split('/')[-1] + '_' + seed + '.pred'
					if os.path.isfile(dev_gold_file) and os.path.isfile(dev_pred_file):
						seed_accuracy, seed_precision, seed_recall, seed_f1, seed_dist, seed_copy = eval(dev_gold_file, dev_pred_file)

						if seed_accuracy is not None:
							all_dev_accuracy.append(seed_accuracy)
							all_dev_precision.append(seed_precision)
							all_dev_recall.append(seed_recall)
							all_dev_f1.append(seed_f1)
							all_dev_dist.append(seed_dist)
							all_dev_copy.append(seed_copy)

					### Evaluating predictions for test file

					test_gold_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.tgt'
					test_filename = test_gold_file.split('.')[0]
					test_pred_file = 'preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_filename.split('/')[-1] + '_' + split + '_' + split_n  + '_' + seed + '.pred'

					if os.path.isfile(test_gold_file) and os.path.isfile(test_pred_file):
						seed_accuracy, seed_precision, seed_recall, seed_f1, seed_dist, seed_copy = eval(test_gold_file, test_pred_file)
						if seed_accuracy is not None:
							all_test_accuracy.append(seed_accuracy)
							all_test_precision.append(seed_precision)
							all_test_recall.append(seed_recall)
							all_test_f1.append(seed_f1)
							all_test_dist.append(seed_dist)
							all_test_copy.append(seed_copy)

				if os.path.isfile(dev_gold_file):
				#	dev_eval_file = io.open('results/' + lg + '/' + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.eval', 'w', encoding = 'utf-8')
				
					if len(all_dev_accuracy) == 3:
						with io.open('results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.eval', 'w', encoding = 'utf-8') as dev_eval_file:
							dev_eval_file.write('Average accuracy: ' + str(round(statistics.mean(all_dev_accuracy), 2)) + ' ' + str(round(statistics.stdev(all_dev_accuracy), 2)) + '\n')
							dev_eval_file.write('Average precision: ' + str(round(statistics.mean(all_dev_precision), 2)) + ' ' + str(round(statistics.stdev(all_dev_precision), 2)) + '\n')
							dev_eval_file.write('Average recall: ' + str(round(statistics.mean(all_dev_recall), 2)) + ' ' + str(round(statistics.stdev(all_dev_recall), 2)) + '\n')
							dev_eval_file.write('Average F1: ' + str(round(statistics.mean(all_dev_f1), 2)) + ' ' + str(round(statistics.stdev(all_dev_f1), 2)) + '\n')
							dev_eval_file.write('Average distance: ' + str(round(statistics.mean(all_dev_dist), 2)) + ' ' + str(round(statistics.stdev(all_dev_dist), 2)) + '\n')
							dev_eval_file.write('Average copy: ' + str(round(statistics.mean(all_dev_copy), 2)) + ' ' + str(round(statistics.stdev(all_dev_copy), 2)) + '\n')
			
					elif len(all_dev_accuracy) == 1:
					#	dev_eval_file.write('Average accuracy: ' + str(round(all_dev_accuracy[0], 2)) + ' ' + '-' + '\n')
					#	dev_eval_file.write('Average precision: ' + str(round(all_dev_precision[0], 2)) + ' ' + '-' + '\n')
					#	dev_eval_file.write('Average recall: ' + str(round(all_dev_recall[0], 2)) + ' ' + '-' + '\n')
					#	dev_eval_file.write('Average F1: ' + str(round(all_dev_f1[0], 2)) + ' ' + '-' + '\n')
					#	dev_eval_file.write('Average distance: ' + str(round(all_dev_dist[0], 2)) + ' ' + '-' + '\n')
					#	dev_eval_file.write('Average copy: ' + str(round(all_dev_copy[0], 2)) + ' ' + '-' + '\n')
						try:
							with io.open('results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.eval', 'w', encoding = 'utf-8') as dev_eval_file:
								dev_eval_file.write('Average accuracy: ' + str(round(all_dev_accuracy[0], 2)) + ' ' + '-' + '\n')
								dev_eval_file.write('Average precision: ' + str(round(all_dev_precision[0], 2)) + ' ' + '-' + '\n')
								dev_eval_file.write('Average recall: ' + str(round(all_dev_recall[0], 2)) + ' ' + '-' + '\n')
								dev_eval_file.write('Average F1: ' + str(round(all_dev_f1[0], 2)) + ' ' + '-' + '\n')
								dev_eval_file.write('Average distance: ' + str(round(all_dev_dist[0], 2)) + ' ' + '-' + '\n')
								dev_eval_file.write('Average copy: ' + str(round(all_dev_copy[0], 2)) + ' ' + '-' + '\n')
						except:
							print('FAILED ' + dev_gold_file)
							print('FAILED ' + dev_pred_file)
							print('\n')
					else:
						try:
							print('REMOVING ' + dev_pred_file)
							os.system('rm ' + dev_pred_file)
						except:
							print('no output')
							print(dev_pred_file)


				if os.path.isfile(dev_gold_file):
					if len(all_test_accuracy) == 3:
						with io.open('results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_test_' + str(i + 1) + '_' + split + '_' + split_n + '.eval', 'w', encoding = 'utf-8') as test_eval_file:
				#			print('writing test eval file')
				#			print(test_eval_file)
							test_eval_file.write('Average accuracy: ' + str(round(statistics.mean(all_test_accuracy), 2)) + ' ' + str(round(statistics.stdev(all_test_accuracy), 2)) + '\n')
							test_eval_file.write('Average precision: ' + str(round(statistics.mean(all_test_precision), 2)) + ' ' + str(round(statistics.stdev(all_test_precision), 2)) + '\n')
							test_eval_file.write('Average recall: ' + str(round(statistics.mean(all_test_recall), 2)) + ' ' + str(round(statistics.stdev(all_test_recall), 2)) + '\n')
							test_eval_file.write('Average F1: ' + str(round(statistics.mean(all_test_f1), 2)) + ' ' + str(round(statistics.stdev(all_test_f1), 2)) + '\n')
							test_eval_file.write('Average distance: ' + str(round(statistics.mean(all_test_dist), 2)) + ' ' + str(round(statistics.stdev(all_test_dist), 2)) + '\n')
							test_eval_file.write('Average copy: ' + str(round(statistics.mean(all_test_copy), 2)) + ' ' + str(round(statistics.stdev(all_test_copy), 2)) + '\n')

					elif len(all_dev_accuracy) == 1:
					#	test_eval_file.write('Average accuracy: ' + str(round(all_test_accuracy[0], 2)) + ' ' + '-'+ '\n')
					#	test_eval_file.write('Average precision: ' + str(round(all_test_precision[0], 2)) + ' ' + '-' + '\n')
					#	test_eval_file.write('Average recall: ' + str(round(all_test_recall[0], 2)) + ' ' + '-' + '\n')
					#	test_eval_file.write('Average F1: ' + str(round(all_test_f1[0], 2)) + ' ' + '-' + '\n')
					#	test_eval_file.write('Average distance: ' + str(round(all_test_dist[0], 2)) + ' ' + '-' + '\n')
					#	test_eval_file.write('Average copy: ' + str(round(all_test_copy[0], 2)) + ' ' + '-' + '\n')
						try:
							with io.open('results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_test_' + str(i + 1) + '_' + split + '_' + split_n + '.eval', 'w', encoding = 'utf-8') as test_eval_file:
								test_eval_file.write('Average accuracy: ' + str(round(all_test_accuracy[0], 2)) + ' ' + '-'+ '\n')
								test_eval_file.write('Average precision: ' + str(round(all_test_precision[0], 2)) + ' ' + '-' + '\n')
								test_eval_file.write('Average recall: ' + str(round(all_test_recall[0], 2)) + ' ' + '-' + '\n')
								test_eval_file.write('Average F1: ' + str(round(all_test_f1[0], 2)) + ' ' + '-' + '\n')
								test_eval_file.write('Average distance: ' + str(round(all_test_dist[0], 2)) + ' ' + '-' + '\n')
								test_eval_file.write('Average copy: ' + str(round(all_test_copy[0], 2)) + ' ' + '-' + '\n')

						except:
							print('FAILED ' + test_gold_file)
							print('FAILED ' + test_pred_file)
							print('\n')

					else:
						try:
							print('REMOVING ' + test_pred_file)
							os.system('rm ' + test_pred_file)
						except:
							print('no output')
							print(test_pred_file)


