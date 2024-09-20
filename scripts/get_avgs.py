import statistics, io, argparse, random, os

def get_avg(filename, metric):
    with open(filename) as f:
        statement = (words for line in f if metric in line for words in line.split())
        # mean is always 3rd word or index 2, followed by std dev
        # print(list(statement)[2])
        return float(list(statement)[2])

parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'language')
parser.add_argument('--test', type = str, default = '0.4', help = 'test set proportion')
parser.add_argument('--model', type = str, help = 'model type')
parser.add_argument('--metric', type = str, help = 'metric to average out')
parser.add_argument('--toolkit', type = str, help = 'toolkit type')
parser.add_argument('--n', type = str, default = '0', help = 'sample size for training splits')


args = parser.parse_args()

lg = args.lg
test_proportion = float(args.test)
model = args.model
metric = args.metric
toolkit = args.toolkit
n = args.n
n_folds = 5 #round(1 / test_proportion)


avg_all_splits = []
for method in ['random']:
    for split in ['random']: # 'adversarial', 'length', 'morph']
        for i in range(n_folds):
            for split_type in ['test']:
                avg_all_seeds = []
                for split_n in ['1', '2', '3']:
                    file = 'results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg + '_' + split_type +'_' + str(i + 1) + '_' + split + '_' + split_n + '.eval'
                    avg_all_seeds.append(get_avg(file, metric))    
                
                mean = statistics.mean(avg_all_seeds)

                avg_all_splits.append(mean)


pooled_mean = statistics.mean(avg_all_splits)
with io.open('results/' + lg + '_' + n + '/' + method + '/test_' + str(int(test_proportion * 100)) + '/' + model + '/' + lg +'_' + toolkit + '_avgs.eval', 'w', encoding = 'utf-8') as avg_f:
    avg_f.write('Averages of 3 Random Seeds for Each Split:\n')
    avg_f.write('test splits ~\n')
    for i in range(n_folds):
        avg_f.write(str(i+1) + ') ' + '{:.3f}'.format(avg_all_splits[i]) + '\n')
    avg_f.write('Average of All train/test Splits:' + '{:.3f}'.format(pooled_mean) + '\n')
