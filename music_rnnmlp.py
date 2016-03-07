from __future__ import print_function

from src.rnnmlp import *

numpy.random.seed(2209)

def trainModel():
#input: nHidden, lr, saveFile
	#model.SimpleRNNModel(nHidden=150, lr=0.01)	
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.train(dataType,sys.argv[3],batch_size=1, num_epoch=120)

def GTrainModel():
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.loadModel(sys.argv[3])	
	model.train(dataType,sys.argv[3],batch_size=1, num_epoch=50)

def generateModel():
	model.LSTMModel(nHidden=int(sys.argv[1]), lr=float(sys.argv[2]))	
	model.loadModel(sys.argv[3])
	chord_file = "./data/MajorChord.json"
	state_file = "./data/LSystem.txt"
	model.generate(24,"sample11.mid",True, 'C', chord_file, state_file, n_steps=80)	
	model.generate(52,"sample22.mid",True, 'C+', chord_file, state_file, n_steps=80)	
	model.generate(57,"sample33.mid",True, 'D+', chord_file, state_file, n_steps=80)	
	
if __name__=='__main__':
	
	model = rnnmlp()
	dataType="Major"

#	trainModel()
#	GTrainModel()
	generateModel()
