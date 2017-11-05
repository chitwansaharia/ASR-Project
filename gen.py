from config import config
import numpy as np
def one_hot(arr):
	final_list = []
	for element in arr:
		temp = [0]*config.config().speaker_reco.num_speakers
		temp[element] = 1
		final_list.append(temp)
	return final_list 



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

	print("Reading done!")
	b = True
	while b:
		curr_list = np.random.permutation(l)[0:BATCH_SIZE]
		curr1 = []
		curr2 = []
		for file in curr_list:
			file_to_open = file[0]
			c = True
			while c:
				try:
					stft = np.load(file_to_open)
					c = False
				except:
					file_to_open = np.random.permutation(l)[0][0]
			
			arr = np.expand_dims(stft['arr_0'])
			mean = mean + arr/341021;

			label = file_to_open.split('/')[8]
			curr1.append(np.expand_dims(stft['arr_0'],2))
			curr2.append(data[label])

			# three_secs = np.split(stft, indices_or_sections = range(300, stft.shape[1], 300), axis=1)
			# curr.append((label, part))
			# print(label, part)
		curr1 = np.stack(curr1,axis=0)
		curr2 = np.stack(one_hot(curr2),axis=0)
		yield((curr1,curr2))
		# b = False

