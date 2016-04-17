import math
import h5py
import numpy as np
import json
import random

class tweet:
	def __init__(self):
		self.token2num={'<pad>':0, '<unk>':1}
		self.num2token={}
		self.width = -1
		self.judge = False
		self.max_len = -1
		self.min_len = 100000
		self.pad = '<pad>'
		self.unk = '<unk>'

	def addToDict(self, oriText):
		self.judge = False
		decText = map(lambda x: x.decode('utf-8'), oriText)
		for line in decText:
			if(len(line)==0):
				continue
			self.max_len = max(self.max_len, len(line)*2)
			self.min_len = min(self.min_len, len(line)*2)
			for char in line:
				if char not in self.token2num:
					self.token2num[char] = len(self.token2num)
		self.cal_width()

	def cal_width(self):
		self.width = int(math.ceil(math.sqrt(len(self.token2num))))

	def cal_num2token(self):
		self.judge = True
		self.num2token = {v: k for k, v in self.token2num.iteritems()}

	def encode(self, input):
		input = input.decode('utf-8')
		result = map(lambda x: [self.token2num[x]%self.width, self.token2num[x]/self.width], input)
		result = [item for sublist in result for item in sublist]
		return result

	def decode(self, oristring):
		if not self.judge:
			self.cal_num2token()
		flag = False
		tmp = 0
		output = u""
		for item in oristring:
			if flag:
				tmp = tmp + item
				output = output + self.num2token[tmp]
			else:
				tmp = item
			flag = not flag
		return output

	def padding(self, oristring, length):
		code = self.token2num[self.pad]
		result = [code for i in range(0, length_pad)]
		idx = 0
		oristring = oristring.decode('utf-8')
		for item in oristring:
			result[idx] = self.token2num[item]%self.width
			result[idx+1] = self.token2num[item]/self.width
			idx = idx+2
		return result

	def padding(self, oristring):
		length = 2*len(oristring)
		length_pad = 0
		for item_len in self.threshold:
			if(length > item_len):
				break
			length_pad = item_len
		return self.padding(oristring, length_pad)

	def process(self, num4padding, val_frac, test_frac):
		with open('fromfemale.txt', 'r') as f:
			fet= f.read().split('\n')
		with open('frommale.txt', 'r') as f:
			mat = f.read().split('\n')
		self.addToDict(fet)
		self.addToDict(mat)
		fet = map(lambda x: x.decode('utf-8'), fet)
		mat = map(lambda x: x.decode('utf-8'), mat)
		self.threshold = [0] * num4padding
		self.threshold[0] = self.max_len - self.min_len
		for i in range(1,num4padding):
			self.threshold[i] = int(self.threshold[i-1]/6)
		self.threshold = [648, 120, 30, 10]# map(lambda x: x + self.min_len, self.threshold)
		#step = (self.max_len- self.min_len)/num4padding
		self.weight_fet = len(mat)/len(fet)
		self.weight_mat = 1
		self.threshold_count = [0, 0, 0, 0] #[0 for v in range(0, num4padding)]
		for line in fet:
			tmplen = 2*len(line)
			for idx in range(1, num4padding+1):
				if(idx==num4padding or tmplen > self.threshold[idx]):
					self.threshold_count[idx-1] += 1
					break
		for line in mat:
			tmplen = 2*len(line)
			for idx in range(1, num4padding+1):
				if(idx==num4padding or tmplen > self.threshold[idx]):
					self.threshold_count[idx-1] += 1
					break
		f = h5py.File('male_female.h5', 'w')
		for idx in range(0, num4padding):
			tmpx = np.empty((self.threshold_count[idx], self.threshold[idx]))
			tmpx.fill(self.token2num[self.pad])
			tmpy = np.empty(self.threshold_count[idx])
			shuffleidx = range(0, self.threshold_count[idx])
			test_idx = self.threshold_count[idx] - int(self.threshold_count[idx]*test_frac)
			val_idx = test_idx - (self.threshold_count[idx]*val_frac)
			random.shuffle(shuffleidx)
			tmpidx = 0
			for line in fet:
				if (idx+1==num4padding or 2*len(line) > self.threshold[idx+1]) and 2*len(line) <= self.threshold[idx]:
					tmptmpidx = 0
					tmpy[shuffleidx[tmpidx]] = 1
					for item in line:
						tmpx[shuffleidx[tmpidx]][tmptmpidx] = self.token2num[item]%self.width
						tmpx[shuffleidx[tmpidx]][tmptmpidx+1] = self.token2num[item]/self.width
						tmptmpidx = tmptmpidx+2
					tmpidx += 1
			for line in mat:
				if (idx+1==num4padding or 2*len(line) > self.threshold[idx+1]) and 2*len(line) <= self.threshold[idx]:
					tmptmpidx = 0
					tmpy[shuffleidx[tmpidx]] = 2
					for item in line:
						tmpx[shuffleidx[tmpidx]][tmptmpidx] = self.token2num[item]%self.width
						tmpx[shuffleidx[tmpidx]][tmptmpidx+1] = self.token2num[item]/self.width
						tmptmpidx = tmptmpidx+2
					tmpidx += 1
			f.create_dataset('x_train'+str(idx+1), data=tmpx[:val_idx])
			f.create_dataset('y_train'+str(idx+1), data=tmpy[:val_idx])
			f.create_dataset('x_val'+str(idx+1), data=tmpx[val_idx:test_idx])
			f.create_dataset('y_val'+str(idx+1), data=tmpy[val_idx:test_idx])
			f.create_dataset('x_test'+str(idx+1), data=tmpx[test_idx:])
			f.create_dataset('y_test'+str(idx+1), data=tmpy[test_idx:])
		json_data = {
			'threshold': self.threshold,
			'threshold_count': self.threshold_count,
			'token2num': self.token2num,
			'width': self.width,
			'weight_fet': self.weight_fet,
			'weight_mat': self.weight_mat,
		}
		with open('male_female.json', 'w') as f:
			json.dump(json_data, f)

	def saveas(self, address):
		with open(address, 'wb') as f:
			json.dump(self.token2num, f)

	def readfrom(self, address):
		with open(address, 'rb') as f:
			self.token2num = json.loads(f.read())
		self.cal_width()
		self.cal_num2token()
