from bs4 import BeautifulSoup
import numpy as np
import itertools as it
import codecs
import constants as c

################################Legacy#####################################
alpha = "abcdefghijklmnopqrstuvwzyx"
#featuerVector = [e for e in it.product(alpha,repeat=k)]
def hash(charList):
	index = 0
	for i in range(0,len(charList)):
		index += (ord(charList[i])-ord('a'))*(26**(len(charList)-i-1))
	return index

def saveSubstring(pointers, inputIndex, story, inputData):
	charList = []
	for p in pointers:
		if story[p] == " ":
			return
		else:
			charList.append(story[p])
	index = hash(charList)
	seqLen = pointers[-1]-pointers[0]
	inputData[inputIndex][index]+=c.lamda**seqLen

def getAllSubstrings(story, index, inputData):
	pointers = [i for i in range(0,c.k)]
	saveSubstring(pointers, index, story, inputData)
	while not(pointers[0] == len(story)-c.k):
		for i in range(0,c.k):
			if i == c.k-1:
				while(pointers[i]<len(story)-1):
					pointers[i]+=1
					saveSubstring(pointers, index, story, inputData)
				if not (pointers[i-1] == pointers[i]-1):
					pointers[i]=pointers[i-1]+2
			else:
				while(pointers[i]<pointers[i+1]-1):
					pointers[i]+=1
					saveSubstring(pointers, index, story, inputData)

def insertInput(found, story, index,inputData, targets):
	getAllSubstrings(story, index, inputData)
	targets[index]=found

def generateFeatureVector(train=True):
	if train:
		samples = c.trainSamples
		path = c.trainPath
	else:
		samples = c.testSamples
		path = c.testPath
	inputIndex = 0
	inputData = np.zeros((samples,26**c.k))
	targets = [0]*samples
	with codecs.open(path,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		for reuter in soup.find_all("reuters"):
				found = 0
				for topic in reuter.topics.find_all("d"):
					if c.label in topic:
						found = 1
						break
				insertInput(found,reuter.body.text,inputIndex, inputData, targets)
				inputIndex+=1
				if inputIndex >= samples:
					break
	return inputData, targets
