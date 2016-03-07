from __future__ import print_function

import glob
import os
import sys
import time

import numpy
import random
import pylab
import json

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from midi.utils import midiread, midiwrite
from keras.layers.core import Dense, TimeDistributedDense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from LSystem import LSystem

def maxProbs(pb):
        pb_sum = numpy.cumsum(pb)
        p = random.random()
        for i in numpy.arange(len(pb_sum)):
                if(pb_sum[i] > p):
                        return i

class rnnmlp ():
        def __init__(self, r=(21, 109), dt=0.3):
                self.r=r
                self.dt=dt
                self.rnnModel = Sequential()
                self.maxFeatures=r[1]-r[0] +1

	'''
	simple RNN model, 
	'''
        def SimpleRNNModel(self, nHidden=120, lr = 0.01):
                self.rnnModel.add(SimpleRNN( nHidden, input_shape =( None, self.maxFeatures), activation='sigmoid', return_sequences=True))
                self.rnnModel.add(TimeDistributedDense(self.maxFeatures))
                self.rnnModel.add(Activation('softmax'))
                rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
                self.rnnModel.compile(loss='categorical_crossentropy', optimizer=rmsprop)

	'''
	LSTM model
	'''
        def LSTMModel(self, nHidden=150, lr = 0.01):
#               print('nHidden: %i\tlr: %.3f' % ( nHidden, lr) )
                self.rnnModel.add(LSTM( nHidden, activation='sigmoid', input_shape =( None, self.maxFeatures), return_sequences=True))
                self.rnnModel.add(TimeDistributedDense(nHidden))
                self.rnnModel.add(Activation('relu'))
                self.rnnModel.add(TimeDistributedDense(self.maxFeatures))
                self.rnnModel.add(Activation('softmax'))
                rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
                self.rnnModel.compile(loss='categorical_crossentropy', optimizer=rmsprop)

        '''
	train module :
	train model , 
	file_name, the name of train or test file
	weight_save_file, save the model parameters
	'''
	def train(self, file_name, weight_save_file, batch_size=1, num_epoch=200):
                print('load data ---------------')

                file_train=os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                'data',file_name,'train','*.mid')
                dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in glob.glob(file_train)]

                file_test=os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                'data',file_name,'test','*.mid')
                testdataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in glob.glob(file_test)]
                print('load done --------------')
                try:
                        for epoch in range(num_epoch):
                                t0 = time.time()
                                numpy.random.shuffle(dataset)
                                costs = []
                                accuracys = []
                                for s, sequence in enumerate(dataset):
                                        y = numpy.hstack((sequence,numpy.zeros((sequence.shape[0],1)) ))
                                        x = numpy.roll(y, 1, axis=0)
                                        x[0,:]=0
                                        x[0,self.maxFeatures-1]=1
                                        cost, accuracy= self.rnnModel.train_on_batch(numpy.array([x]), numpy.array([y]), accuracy=True)
                                        costs.append(cost)
                                        accuracys.append(accuracy)

                                print('epoch: %i/%i\tcost: %.5f\taccu: %.5f\ttime: %.4f s' % (epoch+1, num_epoch, numpy.mean(costs), numpy.mean(accuracys),time.time()-t0))
                                sys.stdout.flush()
                                test_accu=self.evaluate(testdataset)
                                print('test_accu: %.5f' % ( numpy.mean(test_accu)) )
                        self.rnnModel.save_weights(weight_save_file)
                except KeyboardInterrupt:
                        print('interrupt by user !')
	
	'''
	evaluate module :
	evaluate model with test data, compute cost and accuracy
	'''
        def evaluate(self, test_dataset):
                test_accuracy =[]
                for s, sequence in enumerate(test_dataset):
                        test_y = numpy.hstack((sequence,numpy.zeros((sequence.shape[0],1)) ))
                        test_x = numpy.roll(test_y, 1, axis=0)
                        test_x[0,:]=0
                        test_x[0,self.maxFeatures-1]=1
                        cost, accu = self.rnnModel.test_on_batch(numpy.array([test_x]),numpy.array([test_y]), accuracy=True)
                        test_accuracy.append(accu)
                return test_accuracy

	'''
	generate function : 
	generate music or chord,
	init_chord: the first note of the generate sequence
	file_name: file to save the sequence of generate notes
	LS : if true , add Lsystem , generate chord
		if false, no Lsystem, generate music notes
	chord_name: chord name under condition of LS = True
	chord_file: notes of all kinds of chords, load file
	state_file: Lsystem model parameters, load file
	n_steps: the length of generate sequence
	r: notes which counts
	'''
        def generate(self, init_chord, file_name, LS=False, chord_name=None, chord_file=None, state_file=None, n_steps=80, r=(21,109)):
		if(LS):
			Lsystem = LSystem(chord_name, init_chord, chord_file, state_file, r)
                init_sequence = numpy.zeros((1, n_steps +1, self.maxFeatures))
                init_sequence[:, 0, init_chord-self.r[0]] = 1
                for i in numpy.arange(n_steps):
                        probs = self.rnnModel.predict_proba(init_sequence)[:, i, :]
                        for j in numpy.arange(len(init_sequence)):
				if(LS):
					ind = Lsystem.getMaxProbs(probs[j,0:(self.maxFeatures-1)])
				else:
					ind = maxProbs(probs[j,0:(self.maxFeatures-1)])
                                init_sequence[j, i+1, ind] = 1

                generate_sq = [sq[:,0:(self.maxFeatures-1)].nonzero()[1] for sq in init_sequence]
                print(generate_sq[0] + self.r[0])
		print(Lsystem.cur_chord)
                midiwrite(file_name, init_sequence[0,:,0:(self.maxFeatures-1)], self.r, self.dt)



	'''
	load module: load model
	'''
        def loadModel(self, weight_save_file):
                self.rnnModel.load_weights(weight_save_file)

