import json
import numpy
import copy
import random
from BidirecMultiKeyDict import *

def maxProbs(pb):
        pb_sum = numpy.cumsum(pb)
        p = random.random()
        for i in numpy.arange(len(pb_sum)):
                if(pb_sum[i] > p):
                        return i
	return 0

class LSystem():
	def __init__(self, chord_name, chord_num, chord_file, state_file, r=(21,109)):
		self.cur_state = ['A']		#save state, the last is the current
		self.cur_chord = [chord_num]	#save generate chord , the last is the current
		self.cur_opes = ['na','na']	
	
		self.chords = json.load(open(chord_file))	#save the 15 chords
		self.chords = self.chords[chord_name]
		self.r = r
		
		self.operators = BidirectionalDict()	#save state and maping operators
		f=open(state_file,'r')
		for line in f.readlines():
			state,opera=line.strip('\n').split('\t')
			self.operators[state] = opera
		f.close()
	
	'''
	get the next notes by Lsystem and the probability of each notes
	'''
	def getMaxProbs(self, pb, Max=True):
		chords_next, opers_next= self.getNextChords()	#the next probabiliti chord
#		print(opers_next)
		#choose 1 from 3
		maxChord = chords_next[0]
		maxIndex = 0
#		tmp = [ pb[k-self.r[0]] for k in chords_next ]
		if (Max):
#			print (pb)
			for i in range(len(chords_next)):
				if( pb[chords_next[i]-self.r[0]] > pb[maxChord-self.r[0]]):
					maxChord = chords_next[i]
#					print(maxChord)
#					print(i)
					maxIndex = i
#					print(maxIndex)
		else:
			tmp = [ pb[k-self.r[0]] for k in chords_next ]
#			print(tmp)
			maxIndex = maxProbs(tmp)
#			print(maxChord)
#			print(maxIndex)
			maxChord = chords_next[maxIndex]	
		maxOpe = opers_next[maxIndex]
		maxState = copy.deepcopy(self.operators[maxOpe])
		maxState.remove(self.cur_state[-1])
		self.cur_state.append( maxState[0] )			
		self.cur_chord.append( maxChord )
		self.cur_opes.append( maxOpe )
		return maxChord - self.r[0]
	
	'''
	get the next three probable notes by current state,current note
	'''		
	def getNextChords(self):
		cur_state = self.cur_state[-1]
		cur_chord = self.cur_chord[-1]
		oprs = copy.deepcopy(self.operators[cur_state])	#three current operators
		
		cur_index = self.chords.index(cur_chord)
		nextIndexs = []
		
		b=False
		a=False
		print(oprs)
		for i in range(len(oprs)):
			if oprs[i] == "3":
				nextIndexs.append( (cur_index+3)%15 )
			elif oprs[i] == "1":
				nextIndexs.append( (cur_index+1)%15 )
			
			elif oprs[i] == "-1":
				nextIndexs.append( (cur_index-1)%15 )
			elif oprs[i] == "-2":
				nextIndexs.append( (cur_index-2)%15 )
			elif oprs[i] == "#":
				if(self.cur_opes[-2]=="#" and self.cur_opes[-1]=="#"):
					print("b",self.cur_opes[-2:])
					b=True
				else:
					nextIndexs.append( cur_index )
				#nextIndexs.append( (cur_index-3)%15 )
			else: # == "^"
				if(self.cur_opes[-2]=="^" and self.cur_opes[-1]=="^"):
					print("a",self.cur_opes[-2:])
					a=True
				else:
					nextIndexs.append( 15 - 1 - cur_index )
		print(b,a)
		if(b):
			oprs.remove("#")
		if(a):
			oprs.remove("^")
		print( [ self.chords[i] for i in nextIndexs], oprs)
		return [ self.chords[i] for i in nextIndexs], oprs
