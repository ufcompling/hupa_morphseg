import io, os, argparse
import subprocess
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'language')
parser.add_argument('--test', type = str, default = '0.4', help = 'test set proportion')
parser.add_argument('--model', type = str, default = 'transformer', help = 'model architecture')
parser.add_argument('--arch', type = str, default = 'transformer', help = 'model architecture')
parser.add_argument('--batch', type = str, default = '32', help = 'model architecture')
parser.add_argument('--method', type = str, default = 'random', help = 'generating new test samples randomly or adversarially')
parser.add_argument('--n', type = str, default = '0', help = 'sample size for training splits')

args = parser.parse_args()

lg = args.lg
test_proportion = float(args.test)
n_folds = 5 #around(1 / test_proportion)
n = args.n
model = args.model
arch = args.arch
batch_size = args.batch
opt = 'adam'
method = args.method

if not os.path.exists('models/'):
	os.system('mkdir models/')
if not os.path.exists('models/' + lg + '_' + n):
	os.system('mkdir models/' + lg + '_' + n)
if not os.path.exists('models/' + lg + '_' + n + '/random'):
	os.system('mkdir models/' + lg + '_' + n + '/random')
if not os.path.exists('models/' + lg + '_' + n + '/adversarial'):
	os.system('mkdir models/' + lg + '_' + n + '/adversarial')
if not os.path.exists('models/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir models/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('models/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir models/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model)
if not os.path.exists('models/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir models/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('models/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir models/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model)

if not os.path.exists('preds/'):
	os.system('mkdir preds/')
if not os.path.exists('preds/' + lg + '_' + n):
	os.system('mkdir preds/' + lg + '_' + n)
if not os.path.exists('preds/' + lg + '_' + n + '/random'):
	os.system('mkdir preds/' + lg + '_' + n + '/random')
if not os.path.exists('preds/' + lg + '_' + n + '/adversarial'):
	os.system('mkdir preds/' + lg + '_' + n + '/adversarial')
if not os.path.exists('preds/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir preds/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('preds/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir preds/' + lg + '_' + n + '/random/test_' + str(int(test_proportion * 100)) + '/' + model)
if not os.path.exists('preds/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/'):
	os.system('mkdir preds/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/')
if not os.path.exists('preds/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model):
	os.system('mkdir preds/' + lg + '_' + n + '/adversarial/test_' + str(int(test_proportion * 100)) + '/' + model)

DATADIR = r'/blue/liu.ying/n.parkes/hupa_morphseg/data/' + lg + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/'
SAVEDIR = DATADIR + 'checkpoints/' + lg + '-models/'

SRC = lg + '.src'
TGT = lg + '.tgt'
FROMDIR = DATADIR + 'data-bin/'

for split in ['random', 'adversarial']:
	for i in range(n_folds): 
		for split_n in ['1', '2', '3']:
			train_src_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(i + 1) + '_' + split + '_' + split_n + '.src'	
			dev_src_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.src'
			test_src_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.src'

			train_tgt_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_train_' + str(i + 1) + '_' + split + '_' + split_n + '.tgt'	
			dev_tgt_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_dev_' + str(i + 1) + '_' + split + '_' + split_n + '.tgt'
			test_tgt_file = 'data/' + lg + '/' + method + '_' + n + '/test_' + str(int(test_proportion * 100)) + '/' + lg + '_test_' + str(i + 1) + '.tgt'	

			for seed in ['1', '2', '3']:
				if not os.path.exists(DATADIR):
					os.system('mkdir ' + DATADIR)
					os.system('mkdir ' + DATADIR + 'checkpoints/')
					os.system('mkdir ' + SAVEDIR)
				else:
					os.system('rm -r ' + DATADIR)
					os.system('mkdir ' + DATADIR)
					os.system('mkdir ' + DATADIR + 'checkpoints/')
					os.system('mkdir ' + SAVEDIR)	

				if os.path.exists(FROMDIR):
					os.system('rm -r ' + FROMDIR)

				os.system('cp ' + train_src_file + ' ' + DATADIR + 'train.' + lg + '.src')
				os.system('cp ' + train_tgt_file + ' ' + DATADIR + 'train.' + lg + '.tgt')

				os.system('cp ' + dev_src_file + ' ' + DATADIR + 'dev.' + lg + '.src')
				os.system('cp ' + dev_tgt_file + ' ' + DATADIR + 'dev.' + lg + '.tgt')

				os.system('cp ' + test_src_file + ' ' + DATADIR + 'test.' + lg + '.src')
				os.system('cp ' + test_tgt_file + ' ' + DATADIR + 'test.' + lg + '.tgt')

						
				### Preprocessing ###

			#	try:
			#		if 'dict.' + lg + '.src.txt' in os.listdir(DATADIR + '/data-bin'):
			#			os.system('rm ' + DATADIR + '/data-bin/dict*')
			#	except:
			#		pass

				subprocess.call(['fairseq-preprocess',
					 	'--source-lang=' + SRC,
					 	'--target-lang=' + TGT,
					 	'--trainpref='+ DATADIR +'train',
					 	'--validpref='+ DATADIR +'train',
					 	#'--testpref='+TEXTDIR+'test',
					 	'--destdir=' + FROMDIR])
			
				### Training ###

				try:
					os.system('mkdir models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split)
				except:
					pass
				try:
					os.system('mkdir models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1))
				except:
					pass
				try:
					os.system('mkdir models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1) + '/' + split_n)
				except:
					pass
				try:
					os.system('mkdir models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1) + '/' + split_n + '/' + seed)
				except:
					pass

				MODELPATH = 'models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1) + '/' + split_n + '/' + seed + '/checkpoint_best.pt'
				if not os.path.isfile(MODELPATH):
					subprocess.call(['fairseq-train', 
						FROMDIR,
						'--save-dir=' + SAVEDIR,
						'--source-lang=' + SRC,
						'--target-lang=' + TGT,
						'--arch=' + arch,
						'--batch-size=' + batch_size,
					#	'--batch-size-valid=400',
						'--clip-norm=1.0',
						"--criterion=label_smoothed_cross_entropy",
						"--ddp-backend=legacy_ddp",
						'--lr=[0.001]',
						"--lr-scheduler=inverse_sqrt",
						'--max-update=6000',
						'--optimizer=' + opt,
						'--save-interval=10',
						'--patience=10', # early stopping
						'--seed=' + seed,
						'--skip-invalid-size-inputs-valid-test'
						])

					os.system('mv ' + SAVEDIR + 'checkpoint_best.pt models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' +  str(i + 1) + '/' + split_n + '/' + seed + '/')

			### Generating predictions for dev file ###

			for seed in ['1', '2', '3']:
				dev_filename = dev_src_file.split('.')[0].split('/')[-1]
				dev_pred_file = dev_filename + '_' + seed + '.pred'
				if not os.path.isfile('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_pred_file) or os.stat('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_pred_file).st_size == 0:
					print('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_pred_file)
					MODELPATH = 'models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1) + '/' + split_n + '/' + seed + '/checkpoint_best.pt'
					testing = subprocess.Popen(['fairseq-interactive',
						  	FROMDIR,
						  	'--path', MODELPATH,
						  	'--source-lang=' + SRC,
						  	'--target-lang=' + TGT,
						  	'--skip-invalid-size-inputs-valid-test'
						 	],
						 	stdin=open(dev_src_file),
						 	stdout=subprocess.PIPE)				

					with open('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + dev_pred_file, 'w') as f:
						writeline = ''
						testout = testing.stdout.readlines()
						for line in testout:
							line = line.decode('utf-8')
							if line[0] == 'H':
								writeline += line.split('\t')[2]
						f.write(writeline)

			### Generating predictions for test file ###

			for seed in ['1', '2', '3']:
				test_filename = test_src_file.split('.')[0].split('/')[-1]
				test_pred_file = test_filename + '_' + split + '_' + split_n + '_' + seed + '.pred'
				if not os.path.isfile('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_pred_file) or os.stat('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_pred_file).st_size == 0:
					print('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_pred_file)
					MODELPATH = 'models/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + split + '/' + str(i + 1) + '/' + split_n + '/' + seed + '/checkpoint_best.pt'
					testing = subprocess.Popen(['fairseq-interactive',
						  	FROMDIR,
						  	'--path', MODELPATH,
						  	'--source-lang='+SRC,
						  	'--target-lang='+TGT,
						  	'--skip-invalid-size-inputs-valid-test'
						 	],
						 	stdin=open(test_src_file),
						 	stdout=subprocess.PIPE)				

					with open('preds/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + test_pred_file, 'w') as f:
						writeline = ''
						testout = testing.stdout.readlines()
						for line in testout:
							line = line.decode('utf-8')
							if line[0] == 'H':
								writeline += line.split('\t')[2]
						f.write(writeline)

