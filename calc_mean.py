from __future__ import division

from config import config
import numpy as np
import pdb

def gen(mode):
	import numpy as np
	import csv
	import json

	BATCH_SIZE = config.config().speaker_reco.batch_size
	SAMPLE_RATE = 16000
	SNIPPET_SIZE = 3 * SAMPLE_RATE


	# indexfile = sys.argv[1]
	if mode == 'train':
		indexfile = 'train_index'
	else:
		indexfile = 'val_index'
	print(indexfile)
	print("Reading csv file")
	f = open(indexfile, 'r')
	reader = csv.reader(f)
	l = []
	for row in reader:
		l.append(row)

	f.close()
	with open('speaker_names.json') as f:
		data = json.load(f)

	mean,variance = np.zeros((512,300)),np.zeros((512,300))
	count = 0;
	for file in l:
		file_to_open = file[0]
		c = True
		try:
			stft = np.load(file_to_open)
			c = False
			mean += stft['arr_0']
			variance += stft['arr_0']**2/341021
		except:
			continue
		if count%100 == 0:
			print(count*100/len(l))
		count+=1


	mean = mean/341021
	variance = variance - (mean)**2
	mean = np.mean(mean,axis = 1)
	variance = np.mean(variance,axis = 1)

	np.save('normalise.npy',np.matrix([mean,variance]))
		
gen('train')

