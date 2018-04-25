from bs4 import BeautifulSoup
import numpy as np
import itertools as it
import codecs
import constants as c
alpha = "abcdefghijklmnopqrstuvwxyz"

################################Legacy#####################################

def saveSubstring(pointers, story, diction):
	charList = ""
	for p in pointers:
		charList += story[p]
	seqLen = pointers[-1]-pointers[0]
	if charList in diction:
		diction[charList]+=c.lamda**seqLen

def getAllSubstrings(story, index, inputData, features):
	dictionary = {key:0 for key in features}
	pointers = [i for i in range(0,c.k)]
	saveSubstring(pointers, story, dictionary)
	while not(pointers[0] == len(story)-c.k):
		for i in range(0,c.k):
			if i == c.k-1:
				while(pointers[i]<len(story)-1):
					pointers[i]+=1
					saveSubstring(pointers, story, dictionary)
				if not (pointers[i-1] == pointers[i]-1):
					pointers[i]=pointers[i-1]+2
			else:
				while(pointers[i]<pointers[i+1]-1):
					pointers[i]+=1
					saveSubstring(pointers, story, dictionary)
	inputData[index] = [dictionary[feature] for feature in features]

def hash(charList):
	index = 0
	for i in range(0,len(charList)):
		index += (ord(charList[i])-ord('a'))*(26**(len(charList)-i-1))
	return index

def unhash(value):
	charList = ['']*c.k
	for i in range(0,c.k):
		letter = value % 26
		charList[c.k-i-1] = chr(int(letter)+97)
		value = (value-letter)/26
	return ''.join(e for e in charList)


def featureSelection():
	possibleStrings = np.zeros(26**c.k)
	with codecs.open(c.trainPath,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		inputIndex = 0
		for reuter in soup.find_all("reuters"):
			text = reuter.body.text
			for i in range(0,len(text)-c.k):
				if " " in text[i:i+c.k]:
					continue
				possibleStrings[hash(text[i:i+c.k])] += 1
			inputIndex+=1
			if inputIndex >= c.trainSamples:
				break

	c.nFeatures = min(c.nFeatures,len(possibleStrings))
	features = [unhash(index) for index in np.argpartition(possibleStrings,-c.nFeatures)[-c.nFeatures:]]
	return features

def insertInput(found, story, index, inputData, features, targets):
	getAllSubstrings(story, index, inputData, features)
	targets[index]=found

def generateFeatureVector(train=True, features=[]):
	if train:
		samples = c.trainSamples
		path = c.trainPath
	else:
		samples = c.testSamples
		path = c.testPath
	inputIndex = 0
	if(len(features)==0):
		features = featureSelection()
	inputData = np.zeros((samples,c.nFeatures))
	targets = [0]*samples
	with codecs.open(path,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		for reuter in soup.find_all("reuters"):
				found = 0
				for topic in reuter.topics.find_all("d"):
					if c.label in topic:
						found = 1
						break
				insertInput(found,reuter.body.text,inputIndex, inputData, features, targets)
				inputIndex+=1
				if inputIndex >= samples:
					break


	return inputData, targets, features
