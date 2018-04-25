from bs4 import BeautifulSoup
import numpy as np
import itertools as it
import codecs

import preproccess
import sys
import time
import constants as c
sys.setrecursionlimit(2500)

################################Legacy#####################################

alpha = "abcdefghijklmnopqrstuvwxyz"

lamda = c.lamda
k = c.k
targetClass = c.label

stories, sap, targets = preproccess.getData(preproccess.targetClass, c.trainSamples)

currentS = ""
currentT = ""
#-----------helper functions--------------
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

#----------------------------------------------



def dynamicprogk2(s,t,n,sap1,sap2):
	#global sap1,sap2
	#i,endS,endT

	#second auxiliary
	matrixk2 = np.zeros(shape=(n,len(s),len(t)))

	#Initialize K1
	matrixk1 = np.zeros(shape=(n,len(s),len(t)))
	for j in range(len(s)):
		for k in range(len(t)):
			matrixk1[0][j][k]	= 1

	#Fill out K1(and K2)
	for i in range(1,n):
		for j in range(i,len(s)):
			for k in range(i,len(t)):
				#partialsum = 0
				if(matrixk2[i][j][k] == 0):
					#calculate matrixk2[i][j][k]
					if(s[j]==t[k]):
						matrixk2[i][j][k] = lamda*(matrixk2[i][j][k-1]+lamda*matrixk1[i-1][j-1][k-1])
					else:
						for index in sap2[(t,s[j])]:
							if (index >k):break
							matrixk2[i][j][k] += matrixk1[i-1][j][index-1]*(lamda**(k+1-index+2))

						#calculate matrixk2[i][j][k+(til char = s[j])]
						for index in range(len(t)-k):
							if(s[j] != t[k+index]):
								matrixk2[i][j][k+index]= lamda**index*matrixk2[i][j][k]
							else:
								break
				matrixk1[i][j][k]	= lamda*matrixk1[i][j-1][k]+ matrixk2[i][j][k]

	#calculate kn(s,t)
	result = 0
	for j in range(0,len(s)):
		partialsum = 0
		for index in sap2[(t,s[j])]:
			if (index >k):break
			partialsum += matrixk1[n-1][j][index-1]*(lamda**2)
		result+=partialsum

	#Return kn(s,t)
	return result

#Without K2
#Potentially useful
def ssk(s,t):

	sap1 = getStringAlphabetPos([s])
	sap2 = getStringAlphabetPos([t])
	n = c.k
	lamda = c.lamda

	#Initialize K1
	matrixk1 = np.zeros(shape=(n,len(s),len(t)))
	for j in range(len(s)):
		for k in range(len(t)):
			matrixk1[0][j][k]	= 1

	#Fill out K1
	for i in range(1,n):
		for j in range(i,len(s)):
			for k in range(i,len(t)):
				#print((i,j,k))
				partialsum = 0
				for index in sap2[(t,s[j])]:
					if (index >k):break
					partialsum += matrixk1[i-1][j-1][index-1]*(lamda**(k+1-index+2))
				matrixk1[i][j][k]	= lamda*matrixk1[i][j-1][k]+partialsum

	#calculate kn(s,t)
	result = 0
	for j in range(1,len(s)):
		partialsum = 0
		for index in sap2[(t,s[j])]:
			if (index >k):break
			partialsum += matrixk1[n-1][j-1][index-1]*(lamda**2)
		result+=partialsum


	#Return kn(s,t)
	return result


#s and t are the two strings, endS, and endT are end indexes we consider,
#i is the subsequence length
#Auxiliary function to k
#Deprecated
k1cache = {}
def K1(s, t, endS, endT, i):
	global k1cache
	if((endS,endT,i) in k1cache):
		return k1cache[(endS,endT,i)]
	else:
		if(i == 0):
			return 1
		if(min(endS+1,endT+1)<i):
			return 0
		partialsum = 0.0
		for index in sap2[(t, s[endS])]:
			if (index >endT):break
			partialsum+= K1(s,t,endS-1,index-1,i-1)*lamda**(endT+1-index+2)
		#print((endS,endT,i))
		k1cache[(endS,endT,i)] = lamda*K1(s, t, endS-1, endT, i)+partialsum
		return k1cache[(endS,endT,i)]

#s and t are the two strings, endS, and endT are end indexes we consider,
#i is the subsequence length
#Deprecated
kcache = {}
def K(s, t, endS, endT, i):
	global currentS, currentT, kcache
	if(s != currentS and t != currentT):
		kcache.clear()
		k1cache.clear()
		currentS = s
		currentT = t
	if((endS,endT,i) in kcache):
		return kcache[(endS,endT,i)]
	else:
		if(min(endS+1,endT+1)<i):
			#print("gete", min(endS+1,endT+1), i)
			return 0
		partialsum = 0.0
		for index in sap2[(t, s[endS])]:
			#print(index)
			if(index >endT):break
			print("gdfsg")
			partialsum+= K1(s,t,endS-1,index-1,i-1)*lamda**2
		kcache[(endS,endT,i)] = K(s, t, endS-1, endT, i)+partialsum
		return kcache[(endS,endT,i)]

#Contruct the gram matrix for the data
#DataX is a list of strings
def constructGram(Data1, Data2, k,approx = False):
	sap1,sap2 = preproccess.getStringAlphabetPos(Data1) , preproccess.getStringAlphabetPos(Data2)
	start = time.time()
	features = []
	sap3 = {}
	if(approx == True):
		features = featureSelection()
		sap3 = preproccess.getStringAlphabetPos(features)
	gram = np.zeros((len(Data1),len(Data2)))

	#Real gram matrix
	if(approx!=True):
		for i,data1 in enumerate(Data1):
			for j,data2 in enumerate(Data2):
				#print(len(data2))
				print((i,j))
				if(j<len(Data1) and i<len(Data2)):
					if(gram[j][i] != 0):
						gram[i][j] = gram[j][i]
						continue
				gram[i][j] =  dynamicprogk2(data1, data2,k,sap1,sap2)

	#Approximate
	else:
		for i,data1 in enumerate(Data1):
			for j,data2 in enumerate(Data2):
				#print(len(data2))
				print((i,j))
				if(j<len(Data1) and i<len(Data2)):
					if(gram[j][i] != 0):
						gram[i][j] = gram[j][i]
						continue
				approxkernel = [dynamicprogk2(data1,f,k,sap1,sap3)*dynamicprogk2(data2,f,k,sap2,sap3) for f in features]
				gram[i][j] = sum(approxkernel)
	end = time.time()
	print(gram)
	print("time for constructing gram",end - start)
	return gram

stories, sap, targets = getData(preproccess.targetClass, c.trainSamples)
constructGram(stories,stories, k, True)
#dynamicprogk("s","t",1)
