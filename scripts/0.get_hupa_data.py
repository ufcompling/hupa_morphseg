import io, os
import pandas as pd

def remove(character_list):
	if '[' in character_list:
		assert ']' in character_list
		index_list = []
		new_character = ''
		start = [i for i, x in enumerate(character_list) if x == '[']
		end = [i for i, x in enumerate(character_list) if x == ']']
		assert len(start) == len(end)
		if len(list(zip(start, end))) == 1:
			tok = list(zip(start, end))[0]
			new_character += ''.join(c for c in character_list[ : tok[0]])
			new_character += ''.join(c for c in character_list[tok[1] + 1 : ])

		else:
			for i in range(len(list(zip(start, end)))):
				tok = list(zip(start, end))[i]
				try:
					next_tok = list(zip(start, end))[i + 1]
					new_character += ''.join(c for c in character_list[ : tok[0]])
					new_character += ''.join(c for c in character_list[tok[1] + 1 : next_tok[0]])
				except:
					print('no more elements')
			new_character += ''.join(c for c in character_list[list(zip(start, end))[-1][1] + 1: ])

		temp = new_character
		while '!!' in temp:
			temp = temp.replace('!!', '!')

		return list(temp)

	else:
		return character_list


### Clean a bit further

def clean(src_w, tgt_w):

	if type(tgt_w) != float and type(src_w) != float and 'Blue' not in src_w:
		if src_w.startswith("\="):
			src_w = tgt_w[2 : ]

		elif src_w.startswith("\-"):
			src_w = src_w[2 : ]

		elif src_w.startswith("\\"):
			src_w = src_w[1 : ]

		src_w = src_w.replace("\\", "")
		src_w = src_w.replace("=", "")
		src_w = src_w.replace("(", "")
		src_w = src_w.replace(")", "")
		src_w = src_w.replace("-", "")

		if tgt_w.startswith("\="):
			tgt_w = tgt_w[2 : ]

		elif tgt_w.startswith("\-"):
			tgt_w = tgt_w[2 : ]

		elif tgt_w.startswith("\\"):
			tgt_w = tgt_w[1 : ]

		tgt_w = tgt_w.replace("\\", "")
		tgt_w = tgt_w.replace("=", "-")
		tgt_w = tgt_w.replace("(", "")
		tgt_w = tgt_w.replace(")", "")

		tgt_w = tgt_w.split('-') ### '-' indicates morpheme boundary in manual annotations

		seg = '!'.join(m for m in tgt_w)
		if seg[-1] == '!':
			seg = seg[ : -1]

		return src_w, remove(seg)

	return None, None

def gather_data(file):
	data = pd.read_csv(file, encoding = 'utf-8')
	data.dropna()
	orthographic_src_form = data['community orthography'].tolist()
	orthographic_tgt_form = data['parsed community orthography'].tolist()
	phonological_src_form = data['phonemic'].tolist()
	phonological_tgt_form = data['parsed phonemic'].tolist()
	orthographic_cleaned_form, orthographic_cleaned_seg, phonological_cleaned_form, phonological_cleaned_seg = [], [], [], []
	for i in range(len(orthographic_src_form)):
		orthographic_src_w = orthographic_src_form[i]
		orthographic_tgt_w = orthographic_tgt_form[i]
		phonological_src_w = phonological_src_form[i]
		phonological_tgt_w = phonological_tgt_form[i]
		clean_orthographic_src_w, clean_orthographic_seg = clean(orthographic_src_w, orthographic_tgt_w)
		clean_phonological_src_w, clean_phonological_seg = clean(phonological_src_w, phonological_tgt_w)
		if clean_orthographic_src_w is not None and clean_orthographic_seg is not None:
			orthographic_cleaned_form.append(' '.join(c for c in clean_orthographic_src_w if c not in ['[', ']']))
			orthographic_cleaned_seg.append(' '.join(c for c in clean_orthographic_seg if c not in ['[', ']']))		
		if clean_phonological_src_w is not None and clean_phonological_seg is not None:
			phonological_cleaned_form.append(' '.join(c for c in clean_phonological_src_w if c not in ['[', ']']))
			phonological_cleaned_seg.append(' '.join(c for c in clean_phonological_seg if c not in ['[', ']']))
	new_orthographic_cleaned_form, new_orthographic_cleaned_seg, new_phonological_cleaned_form, new_phonological_cleaned_seg = [], [], [], []
	### Getting word types this way (missing one word type in the phonological representations after doing set())
	all_idx = []
	i = 0
	while i < len(orthographic_cleaned_form):
		all_idx.append(i)
		i += 1
	for i in range(len(orthographic_cleaned_form)):
		orthographic_src_w = orthographic_cleaned_form[i]
		orthographic_tgt_w = orthographic_cleaned_seg[i]
		phonological_src_w = phonological_cleaned_form[i]
		phonological_tgt_w = phonological_cleaned_seg[i]

		if orthographic_src_w not in new_orthographic_cleaned_form:
			new_orthographic_cleaned_form.append(orthographic_src_w)
			new_orthographic_cleaned_seg.append(orthographic_tgt_w)
			new_phonological_cleaned_form.append(phonological_src_w)
			new_phonological_cleaned_seg.append(phonological_tgt_w)
	return new_orthographic_cleaned_form, new_orthographic_cleaned_seg, new_phonological_cleaned_form, new_phonological_cleaned_seg

orthographic_cleaned_form, orthographic_cleaned_seg, phonological_cleaned_form, phonological_cleaned_seg = gather_data('resources/Hupa_morphology_data.csv')
with open('hupa-orthography.src', 'w') as f:
	for tok in orthographic_cleaned_form:
		f.write(tok + '\n')
with open('hupa-orthography.tgt', 'w') as f:
	for tok in orthographic_cleaned_seg:
		f.write(tok + '\n')
with open('hupa-phonological.src', 'w') as f:
	for tok in phonological_cleaned_form:
		f.write(tok + '\n')
with open('hupa-phonological.tgt', 'w') as f:
	for tok in phonological_cleaned_form:
		f.write(tok + '\n')

