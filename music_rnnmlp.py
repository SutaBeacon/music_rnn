# coding=utf-8
from __future__ import print_function

from src.rnnmlp import *

numpy.random.seed(2209)

# sys.argv[1], sys.argv[2], sys.argv[3]
# input: nHidden, lr, saveFile

'''
train model 
'''
def trainModel():
	#model.SimpleRNNModel(nHidden=150, lr=0.01)	
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.train(dataType,sys.argv[3],batch_size=1, num_epoch=120)

'''
train model on a trained model
'''
def GTrainModel():
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.loadModel(sys.argv[3])	
	model.train(dataType,sys.argv[3],batch_size=1, num_epoch=50)

'''
generate music or chord
if False, generate music
if True, generate chord, 
'''
def generateModel():
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.loadModel(sys.argv[3])
	chord_file = "./data/MajorChord.json"
	state_file = "./data/LSystem.txt"
	model.generate(28,"./musicResult/sample1_250E.mid",True, 'E', chord_file, state_file, n_steps=80)	
	model.generate(28,"./musicResult/sample2_250E.mid",True, 'Emin', chord_file, state_file, n_steps=80)	
	model.generate(40,"./musicResult/sample3_250E.mid",True, 'Emin+', chord_file, state_file, n_steps=80)	
#	model.generate(31,"./musicResult/sample4_250G.mid",True, 'Gmin', chord_file, state_file, n_steps=80)	
	model.generate(40,"./musicResult/sample5_250E.mid",True, 'Eo', chord_file, state_file, n_steps=80)	
#	model.generate(34,"./musicResult/sample3_250B.mid",True, 'Bb', chord_file, state_file, n_steps=80)	
#	model.generate(62,"./musicResult/sample3_250.mid",n_steps=80)	
	
if __name__=='__main__':
	
	model = rnnmlp()
	dataType="Major"

	trainModel()
#	GTrainModel()
#	generateModel()
