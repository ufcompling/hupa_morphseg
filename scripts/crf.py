import sklearn_crfsuite
import pickle, io, argparse, os


### Gathering data ###

def gather_data(train, dev, test):   # *_tgt files 

	### COLLECT DATA AND LABELLING ###
	train_dict = {}
	dev_dict = {}
	test_dict = {}

	input_files = [train, dev, test] 
	dictionaries = (train_dict, dev_dict, test_dict)

	train_words = []
	dev_words = []
	test_words = []

	counter = 0
#    limit = 0 # init limit
#    n_samples = 1000
	
	for file in input_files:
		data = []

		with io.open(file, encoding = 'utf-8') as f:

			for line in f:
				toks = line.strip().split()
				morphs = (''.join(c for c in toks)).split('!')
				word = ''.join(m for m in morphs)

				if file == train:
					train_words.append(word)

				if file == dev:
					dev_words.append(word)

				if file == test:
					test_words.append(word)

				label = ''
				
				# Binary Labeling:
				# B - bounded (preceding a morpheme bounary) & U - unbounded (not preceding a morpheme boundary)
				label = ''

				if '!' not in toks:
					label = 'U' * len(toks)
				else:
					for i in range(len(toks)-1):
						if toks[i+1] == '!':
							label += 'B'
						elif toks[i] != '!':
							label += 'U'
					label += 'U'

				# Prior Labeling by Beginning, Middle, and End Morpheme Chars
				# for morph in morphs:
				# 	if len(morph) == 1:
				# 		label += 'S'
				# 	else:
				# 		label += 'B'

				# 		for i in range(len(morph)-2):
				# 			label += 'M'

				# 		label += 'E'

				w_dict = {}
				dictionaries[counter][''.join(m for m in morphs)] = label

		counter += 1

	return dictionaries, train_words, dev_words, test_words


### Computing features ###


def features(word_dictonary, original_words, delta):

	X = [] # list (learning set) of list (word) of dics (chars), INPUT for crf
	Y = [] # list (learning set) of list (word) of labels (chars), INPUT for crf
	words = [] # list (learning set) of list (word) of chars

	for word in original_words:
		word_plus = '[' + word + ']' # <w> and <\w> replaced with [ and ]
		word_list = [] # container of the dic of each character in a word
		word_label_list = [] # container of the label of each character in a word
	
		for i in range(len(word_plus)):
			char_dic = {} # dic of features of the actual char
		
			for j in range(delta):
				char_dic['right_' + word_plus[i:i + j + 1]] = 1
		
			for j in range(delta):
				if i - j - 1 < 0: break
				char_dic['left_' + word_plus[i - j - 1:i]] = 1
			char_dic['pos_start_' + str(i)] = 1  # extra feature: left index of the letter in the word
			# char_dic['pos_end_' + str(len(word) - i)] = 1  # extra feature: right index of the letter in the word
		#    if word_plus[i] in ['a', 's', 'o']: # extra feature: stressed characters (discussed in the report)
		#        char_dic[str(word_plus[i])] = 1
			word_list.append(char_dic)
		
			if word_plus[i] == '[': word_label_list.append('[') # labeling start and end
			elif word_plus[i] == ']': word_label_list.append(']')
			else: word_label_list.append(word_dictonary[word][i-1]) # labeling chars

		X.append(word_list)
		Y.append(word_label_list)
		temp_list_word = [char for char in word_plus]
		words.append(temp_list_word)

	return (X, Y, words)


### Building models ###

def build(model_filename, dictionaries, train_words, dev_words, test_words, delta, epsilon, max_iterations):

	train_dict, dev_dict, test_dict = dictionaries

	X_train, Y_train, words_train = features(train_dict, train_words, delta)
	X_dev, Y_dev, words_dev = features(dev_dict, dev_words, delta)
	X_test, Y_test, words_test = features(test_dict, test_words, delta)

	### train ###

#    crf = sklearn_crfsuite.CRF(algorithm = 'ap', epsilon = epsilon, max_iterations = max_iterations)
#    crf.fit(X_train, Y_train, X_dev=X_dev, y_dev=Y_dev)
	
	crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
	)

	crf.fit(X_train, Y_train)

	pickle.dump(crf, io.open(model_filename, "wb"))

	print('training done')

	### Evaluating ###

	Y_dev_predict = crf.predict(X_dev)
	Y_test_predict = crf.predict(X_test)

	return Y_dev_predict, Y_test_predict


def reconstruct(pred_labels, words):

	pred_list = []

	for idx in range(len(pred_labels)):
		pred = pred_labels[idx]
		word = words[idx]

		labels = ''.join(w for w in pred[1 : -1])
		labels = labels.split('E')
	
		if '' in labels:
			labels.remove('')
		new_labels = []

		for tok in labels:
		#    print(tok, word)
			if 'S' not in tok:
				tok += 'E'
				new_labels.append(tok)

			else:
				c = tok.count('S')

				if c == len(tok):
					for z in range(c):
						new_labels.append('S')

				else:
					tok = tok.split('S')

					new_tok = []

					for z in tok:
						if z == '':
							new_labels.append('S')
						else:
							new_labels.append(z + 'E')

		morphs = []

		for i in range(len(new_labels)):
			tok = new_labels[i]

			l = len(tok)

			if i == 0:
				morphs.append(word[0 : l])

			else:
				pre = len(''.join(z for z in new_labels[ : i]))
				morphs.append(word[pre: pre + l])

	#    print(pred, labels, new_labels, word, morphs)

		pred_list.append(morphs)

	return pred_list


parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'language')
parser.add_argument('--test', type = str, default = '0.4', help = 'test set proportion')
parser.add_argument('--d', type = int, default = 4)
parser.add_argument('--e', type = float, default = 0.001)
parser.add_argument('--i', type = int, default = 60)

args = parser.parse_args()

lg = args.lg
test_proportion = float(args.test)
n_folds = 10 #round(1 / test_proportion)

model = 'crf'

if not os.path.exists('models/'):
	os.system('mkdir models/')
if not os.path.exists('models/' + lg):
	os.system('mkdir models/' + lg)
if not os.path.exists('models/' + lg + '/random'):
	os.system('mkdir models/' + lg + '/random')
if not os.path.exists('models/' + lg + '/adversarial'):
	os.system('mkdir models/' + lg + '/adversarial')
if not os.path.exists('models/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir models/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('models/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir models/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/' + model)
if not os.path.exists('models/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir models/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('models/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir models/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model)

if not os.path.exists('preds/'):
	os.system('mkdir preds/')
if not os.path.exists('preds/' + lg):
	os.system('mkdir preds/' + lg)
if not os.path.exists('preds/' + lg + '/random'):
	os.system('mkdir preds/' + lg + '/random')
if not os.path.exists('preds/' + lg + '/adversarial'):
	os.system('mkdir preds/' + lg + '/adversarial')
if not os.path.exists('preds/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir preds/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('preds/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir preds/' + lg + '/random/test_' + str(int(test_proportion * 100)) + '/' + model)
if not os.path.exists('preds/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir preds/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('preds/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir preds/' + lg + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model)

#for method in ['random', 'adversarial']: ## how to generate new test samples
for method in ['random', 'adversarial']:
	for split in ['random', 'adversarial']:#, 'morph', 'length']: ## how to split residual data
		for i in range(n_folds):
			for split_n in ['1', '2', '3']:
				train_file = 'data/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(i + 1) + '_' + split + '_' + split_n + '.tgt'	
				dev_file = 'data/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.tgt'
				test_file = 'data/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.tgt'

				dev_filename = dev_file.split('.')[0]		
				dev_pred_file = 'preds/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_filename.split('/')[-1] + '_1.pred' ## train one random seed for statistcal models

				test_filename = test_file.split('.')[0]
				test_pred_file = 'preds/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_filename.split('/')[-1] + '_' + split + '_' + split_n + '_1.pred'

				if os.path.isfile(train_file) and os.path.isfile(dev_file) and os.path.isfile(test_file):
					model_filename = train_file.split('.')[0].split('/')[-1] + '_1.model' ## train one random seed for statistcal models

					if ((not os.path.isfile(dev_pred_file)) or os.stat(dev_pred_file).st_size == 0) or ((not os.path.isfile(test_pred_file)) or os.stat(test_pred_file).st_size == 0):
						dictionaries, train_words, dev_words, test_words = gather_data(train_file, dev_file, test_file)

						Y_dev_predict, Y_test_predict = build('models/' + lg + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + model_filename, dictionaries, train_words, dev_words, test_words, args.d, args.e, args.i)

						dev_predictions = reconstruct(Y_dev_predict, dev_words)				
		
						with io.open(dev_pred_file, 'w', encoding = 'utf-8') as f:
							for tok in dev_predictions:
								tok = '!'.join(m for m in tok)
								tok = list(tok)
								f.write(' '.join(c for c in tok) + '\n')

						test_predictions = reconstruct(Y_test_predict, test_words)
					
						with io.open(test_pred_file, 'w', encoding = 'utf-8') as f:
							for tok in test_predictions:
								tok = '!'.join(m for m in tok)
								tok = list(tok)
								f.write(' '.join(c for c in tok) + '\n')

				else:
					print(train_file)
					print(dev_file, dev_pred_file)
					print(test_file, test_pred_file)




