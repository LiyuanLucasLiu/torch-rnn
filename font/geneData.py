import h5py
import argparse, json, os
import numpy as np
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default = '../data/wiki0.txt')
parser.add_argument('--input_font', default = './dict.np')
parser.add_argument('--output_h5', default = '../data/wiki0.h5')
parser.add_argument('--output_json', default = '../data/wiki0.js')
parser.add_argument('--val_frac', type = float, default = 0.01)
parser.add_argument('--test_frac', type = float, default = 0.01)
parser.add_argument('--train_frac', type = float, default = 0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default = 'utf-8')
args = parser.parse_args()

with open('CharTable.json', 'rb') as f:
	ct = json.loads(f.read())
cd = {}
for idx in range(0, len(ct)):
	cd[ct[idx]] = idx

with open(args.input_font, 'rb') as f:
	font = np.load(f)

total_size = 0
with codecs.open(args.input_txt, 'r', args.encoding) as f:
	for line in f:
		total_size += len(line)

val_size = int(args.val_frac * total_size)
test_size = int(args.test_frac * total_size)
train_size = int(args.train_frac * total_size)

if not args.quiet:
	print 'Training size: %d' % train_size
	print 'Val Size: %d' % val_size
	print 'Test Size: %d' % test_size

dtype = np.uint8
train = np.zeros((train_size, font.shape[1], font.shape[2]), dtype = dtype)
test = np.zeros((test_size, font.shape[1], font.shape[2]), dtype = dtype)
val = np.zeros((val_size, font.shape[1], font.shape[2]), dtype = dtype)
splits = [train, val, test]

split_idx, cur_idx = 0, 0
with codecs.open(args.input_txt, 'r', args.encoding) as f:
	for line in f:
		for char in line:
			if char not in cd:
				splits[split_idx][cur_idx] = font[cd[u'\n']]
			else:	
				splits[split_idx][cur_idx] = font[cd[char]]
			cur_idx += 1
			if cur_idx == splits[split_idx].shape[0]:
				split_idx += 1
				cur_idx = 0
			if split_idx == 3:
				break
		if split_idx == 3:
			break

with h5py.File(args.output_h5, 'w') as f:
	f.create_dataset('train', data=train)
	f.create_dataset('val', data=val)
	f.create_dataset('test', data=test)
	f.create_dataset('font', data=font)

json_data = {
	'cd':cd,
	'ct':ct,
}
with open(args.output_json, 'w') as f:
	json.dump(json_data, f)
