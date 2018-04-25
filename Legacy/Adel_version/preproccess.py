from bs4 import BeautifulSoup
import numpy as np
import itertools as it
import codecs
import constants as c
lamda = c.lamda
k = c.k
targetClass = c.label

alpha = "abcdefghijklmnopqrstuvwxyz"

def hash(char):
	return ord(char)-ord('a')

def getStories(targetClass, samples=20, path="train.txt"):
	stories = [""]*samples
	targets = [0]*samples
	inputIndex = 0
	with codecs.open(path,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		for reuter in soup.find_all("reuters"):
				found = 0
				for topic in reuter.topics.find_all("d"):
					if targetClass in topic:
						found = 1
						break
				story = reuter.body.text
				stories[inputIndex] = story
				targets[inputIndex] = found
				inputIndex+=1
				if inputIndex >= samples:
					break
	return stories, targets


#Get position of all charachters in all stories
#Meant to be used once then saved
def getStringAlphabetPos(stories):

	#Initialize
	stringAlphabetPos = {}
	for story in stories:
		for char in alpha:
			stringAlphabetPos[(story, char)] = []
		stringAlphabetPos[(story, ' ')] = []

	#Fill out dictionary with indexes
	for story in stories:
		for i, char in enumerate(story):
			if(char == ' '): continue
			stringAlphabetPos[(story, char)].append(i)

	return stringAlphabetPos


#Return the stories, their character positions, and the targets
def getData(targetClass, samples=5, path="train.txt"):
	stories, targets, = getStories(targetClass, samples, path)
	stringAlphabetPos = getStringAlphabetPos(stories)
	return stories, stringAlphabetPos, targets


#stringsk = [e for e in it.product(alpha,repeat=k)]
#stories, targets = getStories(targetClass)
#storySubseqPos = {}


'''
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
	global lamda
	inputData[inputIndex][index]+=lamda**seqLen

def getAllSubstrings(story, index, inputData):
	pointers = [i for i in range(0,k)]
	saveSubstring(pointers, index, story, inputData)
	while not(pointers[0] == len(story)-k):
		for i in range(0,k):
			if i == k-1:
				while(pointers[i]<len(story)-1):
					pointers[i]+=1
					saveSubstring(pointers, index, story, inputData)
				if not (pointers[i-1] == pointers[i]-1):
					pointers[i]=pointers[i-1]+2
			else:
				while(pointers[i]<pointers[i+1]-1):
					pointers[i]+=1
					saveSubstring(pointers, index, story, inputData)


#Insert data data into inputdata
def insertInput(label, story, index,inputData, targets):
	getAllSubstrings(story, index, inputData)
	targets[index]=label

def generateFeatureVector(targetClass, samples=20, path="train.txt"):
	inputIndex = 0
	inputData = np.zeros((samples,26**k))
	targets = [0]*samples
	with codecs.open(path,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		for reuter in soup.find_all("reuters"):
				found = 0
				for topic in reuter.topics.find_all("d"):
					if targetClass in topic:
						found = 1
						break
				insertInput(found,reuter.body.text,inputIndex, inputData, targets)
				inputIndex+=1
				if inputIndex >= samples:
					break
	return inputData, targets



#Get featurevector for one string
#def getFeatureVector(story, k):
#	inputData = np.zeros((1,26**k))
#	getAllSubstrings(story, 0, inputData)
#	return inputData[0]
'''
#stories, sap, targets = getData(targetClass);
#print stories[0]
#print sap[(stories[0], 'j')]
