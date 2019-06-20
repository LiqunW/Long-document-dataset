import os
import numpy as np
import _pickle as cPickle
import re

from config import Config
config = Config()
import time

class Dataloader(object):
	def shuffle(self, l_file,l_label):
		l=list(zip(l_file,l_label))
		np.random.shuffle(l)
		s_file,s_label = zip(*l)
		return list(s_file),list(s_label)

	# Assume dataset is organized in a folder "path"
	# path/something1
	# path/something2
	# path/.....
	# path/somethingK
	# Then we have K classes, so we will have a map corresponding to [1,something1], [2, something2], ..., [K, somethingK]
	def __init__(self, path, batchsize):
		folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path,f))]
		iClass = 0
		self.dictCls2Idx = {}
		self.Dataset = []
		self.Labels  = []
		self.nBatchsize = batchsize

		for sub in folders:
			iClass = iClass + 1
			subfolder = os.path.join(path,sub)
			self.dictCls2Idx[sub] = iClass
			self.dictCls2Idx[iClass] = sub

			files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder,f))]

			for f in files:
				fname = os.path.join(subfolder,f)
				self.Dataset.append(fname)
				self.Labels.append(iClass)

		self.nTotal = len(self.Dataset)
#如果是第一次运行dataloader，就会创建一个文本，使dataloader固定下来
		try:
			Dataset_file = open('Dataset.txt','rb')
			Labels_file = open('Labels_file.txt', 'rb')
		except:
			print('Dataset has not been created,creating Dataset')
			self.Dataset, self.Labels = self.shuffle(self.Dataset, self.Labels)
			with open('Dataset.txt','wb') as Dataset_file:
				cPickle.dump(self.Dataset,Dataset_file)
			with open('Labels_file.txt', 'wb') as Labels_file:
				cPickle.dump(self.Labels,Labels_file)
			print('Dataset created')
		else:
			self.Dataset = cPickle.load(Dataset_file)
			Dataset_file.close()
			self.Labels = cPickle.load(Labels_file)
			Labels_file.close()
		# 80% for training, 10% for validation, 10% for testing
		self.nTrain = int(self.nTotal*0.8)
		self.trainDataset = self.Dataset[0 : self.nTrain]
		self.trainLabels = self.Labels[0 : self.nTrain]

		self.nVal = int(self.nTotal*0.1)
		self.valDataset = self.Dataset[self.nTrain : self.nTrain+self.nVal]
		self.valLabels = self.Labels[self.nTrain : self.nTrain+self.nVal]

		self.nTest = self.nTotal - self.nTrain - self.nVal
		self.testDataset = self.Dataset[self.nTrain+self.nVal : self.nTotal]
		self.testLabels = self.Labels[self.nTrain + self.nVal: self.nTotal]

		self.train_pos = 0
		self.val_pos = 0
		self.test_pos = 0

	def next_train_batch(self):
		bEnd = False
		output_files = []
		output_labels = []
		documents = np.zeros([self.nBatchsize, config.num_word],dtype=np.int32)
		if self.train_pos + self.nBatchsize < self.nTrain:
			output_files = self.trainDataset[self.train_pos : self.train_pos + self.nBatchsize]
			output_labels=self.trainLabels[self.train_pos : self.train_pos + self.nBatchsize]
			self.train_pos = self.train_pos + self.nBatchsize
			i=0
			for out_file in output_files:
				documents[i, :] = np.load(out_file)
				i=i+1
			output_labels = np.asarray(output_labels) - 1 #把[1-10]的下标变为[0,9]
		elif self.train_pos + self.nBatchsize > self.nTrain and self.train_pos < self.nTrain:
			output_files = self.trainDataset[self.train_pos: self.nTrain]
			output_labels = self.trainLabels[self.train_pos: self.nTrain]
			output_files_1 = self.trainDataset[0:self.nBatchsize-(self.nTrain-self.train_pos)]
			output_labels_1 = self.trainLabels[0:self.nBatchsize-(self.nTrain-self.train_pos)]
			output_files = output_files + output_files_1
			output_labels = output_labels + output_labels_1
			i=0
			for out_file in output_files:
				documents[i,:] = np.load(out_file)
				i=i+1
			output_labels = np.asarray(output_labels) - 1
			self.train_pos = self.train_pos + self.nBatchsize
		else:
			self.trainDataset, self.trainLabels = self.shuffle(self.trainDataset, self.trainLabels)
			self.train_pos = 0
			bEnd = True
		return documents, output_labels, bEnd, output_files

# noinspection PyUnreachableCode
	def next_val_batch(self):
			bEnd = False
			output_files = []
			output_labels = []
			documents = np.zeros([self.nBatchsize, config.num_word],dtype=np.int32)
			if self.val_pos + self.nBatchsize < self.nVal:
				output_files = self.valDataset[self.val_pos : self.val_pos + self.nBatchsize]
				output_labels=self.valLabels[self.val_pos : self.val_pos + self.nBatchsize]
				self.val_pos = self.val_pos + self.nBatchsize
				i=0
				for out_file in output_files:
					documents[i, :] = np.load(out_file)
					i=i+1
				output_labels = np.asarray(output_labels) - 1
			elif self.val_pos + self.nBatchsize > self.nVal and self.val_pos < self.nVal:
				output_files = self.valDataset[self.val_pos: self.nVal]
				output_labels = self.valLabels[self.val_pos: self.nVal]
				output_files_1 = self.valDataset[0:self.nBatchsize - (self.nVal - self.val_pos)]
				output_labels_1 = self.valLabels[0:self.nBatchsize - (self.nVal - self.val_pos)]
				output_files = output_files + output_files_1
				output_labels = output_labels + output_labels_1
				i = 0
				for out_file in output_files:
					documents[i, :] = np.load(out_file)
					i = i + 1
				output_labels = np.asarray(output_labels) - 1
				self.val_pos = self.val_pos + self.nBatchsize
			else:
				self.valDataset, self.valLabels = self.shuffle(self.valDataset, self.valLabels)
				self.val_pos = 0
				bEnd = True
			return documents, output_labels, bEnd, output_files


	def next_test_batch(self):
			bEnd = False
			output_files = []
			output_labels = []
			documents = np.zeros([self.nBatchsize,config.num_word],dtype=np.int32)
			if self.test_pos + self.nBatchsize < self.nTest:
				output_files = self.testDataset[self.test_pos:self.test_pos+self.nBatchsize]
				output_labels = self.testLabels[self.test_pos:self.test_pos+self.nBatchsize]
				self.test_pos = self.test_pos + self.nBatchsize
				i = 0
				for out_file in output_files:
					documents[i,:] = np.load(out_file)
					i = i + 1
				output_labels = np.asarray(output_labels) - 1
			elif self.test_pos + self.nBatchsize > self.nTest and self.test_pos < self.nTest:
				output_files = self.testDataset[self.test_pos: self.nTest]
				output_labels = self.testLabels[self.test_pos: self.nTest]
				output_files_1 = self.testDataset[0:self.nBatchsize - (self.nTest - self.test_pos)]
				output_labels_1 = self.testLabels[0:self.nBatchsize - (self.nTest - self.test_pos)]
				output_files = output_files + output_files_1
				output_labels = output_labels + output_labels_1
				i = 0
				for out_file in output_files:
					documents[i, :] = np.load(out_file)
					i = i + 1
				output_labels = np.asarray(output_labels) - 1
				self.test_pos = self.test_pos + self.nBatchsize
			else:
				self.testDataset,self.testLabels = self.shuffle(self.testDataset,self.testLabels)
				self.test_pos = 0
				bEnd = True
			return documents,output_labels,bEnd,output_files

	def get_class_nameOridx(self, name):
		return self.dictCls2Idx[name]

