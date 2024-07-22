mkdir data
mkdir data/hupa
mkdir data/hupa/original

python3 scripts/0.get_hupa_data.py  # This generates *.src and *.tgt files for both orthographic and phonetic representations

mv hupa-* data/hupa/original  # Move the generated data to this folder

# This will print out some output, which you can ignore for now
# --test refers to test set proportion, setting to 0.2, but later we can see if we want to experiment with different proportions
python3 scripts/1.\ data_split.py --lg hupa --test 0.2 --method random  

python3 scripts/crf.py --lg hupa --test 0.2