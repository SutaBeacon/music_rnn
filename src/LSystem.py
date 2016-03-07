import json
import numpy
import copy
from BidirecMultiKeyDict import *
class LSystem():
	def __init__(self, chord_name, chord_num, chord_file, state_file, r=(21,109)):
		self.cur_state = ['A']		#save state, the last is the current
		self.cur_chord = [chord_num]	#save generate chord , the last is the current

		self.chords = json.load(open(chord_file))	#save the 15 chords
		self.chords = self.chords[chord_name]
		self.r = r
		
		self.operators = BidirectionalDict()	#save state and maping operators
		f=open(state_file,'r')
		for line in f.readlines():
			state,opera=line.strip('\n').split('\t')
			self.operators[state] = opera
		f.close()

	def getMaxProbs(self, pb, Max=True):
		chords_next, opers_next= self.getNextChords()	#the next probabiliti chord

		#choose 1 from 3
		maxChord = chords_next[0]
		maxIndex = 0
		if(Max):
			for i in range(len(chords_next)):
				if( pb[chords_next[i]-self.r[0]] > pb[maxChord-self.r[0]]):
					maxChord = chords_next[i]
					maxInedx = i
		maxOpe = opers_next[maxIndex]
		maxState = copy.deepcopy(self.operators[maxOpe])
		maxState.remove(self.cur_state[-1])
		self.cur_state.append( maxState[0] )			
		self.cur_chord.append( maxChord )
		return maxChord - self.r[0]
			
	def getNextChords(self):
		cur_state = self.cur_state[-1]
		cur_chord = self.cur_chord[-1]
		oprs = self.operators[cur_state]	#three current operators
		
		cur_index = self.chords.index(cur_chord)
		nextIndexs = []
		
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
				nextIndexs.append( cur_index )
			else: # == "^"
				nextIndexs.append( 15 - 1 - cur_index )
		
		return [ self.chords[i] for i in nextIndexs], oprs
