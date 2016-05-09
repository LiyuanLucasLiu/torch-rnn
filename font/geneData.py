import h5py
import argparse, json, os
import numpy as np
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default = '../data/wiki0.txt')
parser.add_argument('--input_font', default = './dict.np')
parser.add_argument('--output_h5', default = '../data/wiki0.h5')
parser.add_argument('--output_font', default = '../data/font.h5')
parser.add_argument('--output_json', default = '../data/wiki0.js')
parser.add_argument('--val_frac', type = float, default = 0.003) 
parser.add_argument('--test_frac', type = float, default = 0.003)
parser.add_argument('--train_frac', type = float, default = 0.03)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default = 'utf-8')
parser.add_argument('--threshold', default = 50)
args = parser.parse_args()

with open('CharTable.json', 'rb') as f:
	ct = json.loads(f.read())
cd = {}
cc = {}
for idx in range(0, len(ct)):
	cd[ct[idx]] = idx
	cc[ct[idx]] = 0

with open(args.input_font, 'rb') as f:
	font = np.load(f)

total_size = 0
with codecs.open(args.input_txt, 'r', args.encoding) as f:
	for line in f:
		total_size += len(line)

val_size = int(args.val_frac * total_size)
test_size = int(args.test_frac * total_size)
train_size = int(args.train_frac * total_size)
total_size = val_size + test_size+ train_size
if not args.quiet:
	print 'Total Size: %d' % total_size
	print 'Training Size: %d' % train_size
	print 'Val Size: %d' % val_size
	print 'Test Size: %d' % test_size

cur_idx = 0
with codecs.open(args.input_txt, 'r', args.encoding) as f:
	for line in f:
		for char in line:
			cur_idx += 1
			if char in cc:
				cc[char] = cc[char] + 1
			if cur_idx == total_size:
				break
		if cur_idx == total_size:
			break

cur_idx = 2
for k, v in cc.iteritems():
	if v <= args.threshold:
		cc[k] = 1
	else:
		cc[k] = cur_idx
		cur_idx = cur_idx + 1
char_set_size = cur_idx

dtype = np.uint8
dtype2 = np.uint32
train = np.zeros((train_size, font.shape[1], font.shape[2]), dtype = dtype)
test = np.zeros((test_size, font.shape[1], font.shape[2]), dtype = dtype)
val = np.zeros((val_size, font.shape[1], font.shape[2]), dtype = dtype)
train_1d = np.zeros(train_size, dtype = dtype2)
test_1d = np.zeros(test_size, dtype = dtype2)
val_1d = np.zeros(val_size, dtype = dtype2)
splits = [train, val, test]
splits_1d = [train_1d, val_1d, test_1d]

split_idx, cur_idx = 0, 0
with codecs.open(args.input_txt, 'r', args.encoding) as f:
	for line in f:
		for char in line:
			code = ord(char)
			if code ==  0x0020:
				code = 0x3000
			if code > 0x0020 and code <= 0x007e:
				code += 0xfee0
			char = unichr(code)
			if char not in cd:
				# print(char)
				# print(code)
				# raw_input()
				splits[split_idx][cur_idx] = font[cd['?']]
				splits_1d[split_idx][cur_idx] = 1
			else:	
				splits[split_idx][cur_idx] = font[cd[char]]
				splits_1d[split_idx][cur_idx] = cc[char]
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
	f.create_dataset('train_1d', data=train_1d)
	f.create_dataset('val_1d', data=val_1d)
	f.create_dataset('test_1d', data=test_1d)
	f.create_dataset('font', data=font)

with h5py.File(args.output_font, 'w') as f:
	f.create_dataset('font', data = font)

rcc = { v:k for k, v in cc.iteritems()}

json_data = {
	'cd':cd,
	'char_set_size':char_set_size,
	'cc':cc,
	'rcc':rcc
}
with open(args.output_json, 'w') as f:
	json.dump(json_data, f)
