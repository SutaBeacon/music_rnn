from __future__ import print_function

import glob
import os
import sys

import numpy
import pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from midi.utils import midiread, midiwrite
from keras.layers.core import Dense, TimeDistributedDense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.optimizers import SGD
numpy.random.seed(2209)


def maxProbs(pb):
	maxP = pb[0]
	maxIndex=0
	for i in numpy.arange(len(pb)):
		if pb[i] >pb[maxIndex]:
			maxIndex=i
	return i
	
		
class rnnmlp ():
	def __init__(self, nHidden=150, lr = 0.01, r=(0, 128), dt=0.3):

#		print('loadData ---------------')
#		assert len(files) > 0, 'Training set is empty!'
#		dataset = [midiread(f, self.r, self.dt).paino_roll.astype(theano.config.floatX) for f in files]
#		print('load done --------------')
#		print('process data -----------')
#		numpy.random.shuffle(dataset)
#		y = []
#		x = []
#		for i, sequence in enumerate(dataset):
#			y.append(numpy.hstack((sequence,numpy.zeros((sequence.shape[0],1)) )) )
#			tmp = numpy.roll(y, 1, axis=0)
#			tmp[0,:]=0
#			tmp[0,maxFeatures-1]=1
#			x.append(tmp)
			
		self.r=r
		self.dt=dt	
		self.rnnModel = Sequential()
		self.rnnModel.add(SimpleRNN( nHidden, input_shape =( None, maxFeatures), activation='relu', return_sequences=True))
		self.rnnModel.add(TimeDistributedDense(maxFeatures))
		self.rnnModel.add(Activation('softmax'))
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.rnnModel.compile(loss='categorical_crossentropy', optimizer=sgd)
	
	def train(self, files, weightSaveFile, batchSize=1, numEpoch=200):
		print('loadData ---------------')
		assert len(files) > 0, 'Training set is empty!'
		dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in files]
		print('load done --------------')
		try:
			for epoch in range(numEpoch):
				numpy.random.shuffle(dataset)
				costs = []
				for s, sequence in enumerate(dataset):
					y = numpy.hstack((sequence,numpy.zeros((sequence.shape[0],1)) ))
					x = numpy.roll(y, 1, axis=0)
					x[0,:]=0
					x[0,maxFeatures-1]=1
					cost = self.rnnModel.train_on_batch(numpy.array([x]), numpy.array([y]) )
					costs.append(cost)

				print('epoch: %i/%i' % (epoch+1, numEpoch))
				print(numpy.mean(costs))
				sys.stdout.flush()
			rnnModel.save_weight(weightSaveFile)
		except KeyboardInterrupt:
			print('interrupt by user !')

	def generate(self, filename, nSteps=80 ,show=False):
		init_sequence = numpy.zero((1, nSteps +1, maxFeatures))
		init_sequence[:, 0, maxFeatures-1] = 1
		for i in numpy.arange(nSteps):
			probs = self.rnnModel.predict_proba(init_sequence, batchSize=1)[:, i, :]
			for j in numpy.arrange(len(init_sequence)):
				init_sequence[j, i+1, maxProbs(probs[j,:])] = 1

		generate_sq = [sq[1:].nonzero()[1] for sq in init_sequence] 
		midiwrite(filename, generate_sq, self.r, self.dt)

	def loadModel(weightFile):
		rnn_model.load_weights("weight_file.h5")

if __name__=='__main__':

	maxFeatures = 128 + 1
	model = rnnmlp();
	
	file_path=os.path.join(os.path.split(os.path.dirname(__file__))[0],
				'data','Nottingham','train','*.mid')
	model.train(glob.glob(file_path),'weight_file.h5',batchSize=1, numEpoch=200)
	
