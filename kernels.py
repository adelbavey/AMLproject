import codecs
import numpy as np
import constants as c
from bs4 import BeautifulSoup
import functools
import time
import operator
import multiprocessing as mp


def getTexts(train=True):
	if train:
		samples = c.trainSamples
		path = c.trainPath
	else:
		samples = c.testSamples
		path = c.testPath
	ratio = samples/c.labelconsts[c.label]
	inputIndex = 0
	targets = [0]*samples
	posSamples = 0
	texts = [""]*samples
	with codecs.open(path,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		for reuter in soup.find_all("reuters"):
				found = 0
				valid = False
				for topic in reuter.topics.find_all("d"):
					for white in c.labels:
						if white in topic:
							valid = True
					if c.label in topic:
						found = 1
						break
				if valid:
					body = reuter.body.text
					if(len(body) < 1000 and ((found and posSamples<ratio) or ((not found) and posSamples>=ratio))):
						texts[inputIndex]=body
						targets[inputIndex]=found
						inputIndex+=1
						posSamples += found
				if inputIndex >= samples:
					break
	return texts,targets

def featureSelection():
	possibleStrings = {}
	with codecs.open(c.trainPath,"r", encoding="utf-8", errors="replace") as trainFile:
		soup = BeautifulSoup(trainFile.read(), 'html.parser')
		inputIndex = 0
		for reuter in soup.find_all("reuters"):
			text = reuter.body.text
			for i in range(0,len(text)-c.k):
				s = text[i:i+c.k]
				if " " in s:
					continue
				if s in possibleStrings:
					possibleStrings[s]+=1
				else:
					possibleStrings[s]=1
			inputIndex+=1
			#if inputIndex >= 80:
			#	break
	c.nFeatures = min(c.nFeatures,len(possibleStrings))
	return [x[0] for x in sorted(possibleStrings.items(), key=operator.itemgetter(1))[-c.nFeatures:]]

@functools.lru_cache(maxsize=None)
def ngram(x,s):
	total = 0
	for i in range(0,len(x)-c.k):
		for j in range(0,len(s)-c.k):
			if x[i:i+c.k] == s[j:j+c.k]:
				total += 1
	return total

@functools.lru_cache(maxsize=None)
def bag(x,s):
	bagowords = {}
	for word in x.split():
		if word not in bagowords:
			bagowords[word] = [0,0]
		bagowords[word][0] += 1
	for word in s.split():
		if word not in bagowords:
			bagowords[word] = [0,0]
		bagowords[word][1] += 1
	total = 0
	for word in bagowords:
		total += bagowords[word][0]*bagowords[word][1]
	return total


#Get position of all charachters in all stories
#Meant to be used once then saved
def getStringAlphabetPos(stories):

	#Initialize
	stringAlphabetPos = {}
	for story in stories:
		for char in c.alpha:
			stringAlphabetPos[(story, char)] = []
		stringAlphabetPos[(story, ' ')] = []

	#Fill out dictionary with indexes
	for story in stories:
		for i, char in enumerate(story):
			if(char == ' '): continue
			stringAlphabetPos[(story, char)].append(i)

	return stringAlphabetPos

@functools.lru_cache(maxsize=None)
def ssk(s,t):

	sap1 = getStringAlphabetPos([s])
	sap2 = getStringAlphabetPos([t])
	n = c.k
	lamda = c.lamda

	#second auxiliary
	matrixk2 = np.zeros(shape=(n,len(s),len(t)))

	#Initialize K1
	matrixk1 = np.zeros(shape=(n,len(s),len(t)))
	for j in range(len(s)):
		for k in range(len(t)):
			matrixk1[0][j][k] = 1

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
							matrixk2[i][j][k] += matrixk1[i-1][j-1][index-1]*(lamda**(k+1-index+2))

						#calculate matrixk2[i][j][k+(til char = s[j])]
						for index in range(len(t)-k):
							if(s[j] != t[k+index]):
								matrixk2[i][j][k+index]= (lamda**index)*matrixk2[i][j][k]
							else:
								break
				matrixk1[i][j][k]	= lamda*matrixk1[i][j-1][k]+ matrixk2[i][j][k]

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

def kernel(x,s,f):
	return f(x,s)/(np.sqrt(f(x,x)*f(s,s)))

def approxkernel(x,s,f,features):
	head = left = right = 0.0
	for si in features:
		head+=f(x,si)*f(s,si)
		left+=f(x,si)**2
		right+=f(s,si)**2
	if head == 0.0:
		return 0.0
	return head/(np.sqrt(left*right))

def calculateMatrix(texts1, texts2,f, approx,features):
	pool = mp.Pool()
	poolDic = {}
	for i in range(0,len(texts1)):
		for j in range(0,len(texts2)):
			if approx:
				poolDic[(i,j)] = pool.apply_async(approxkernel, args = (texts1[i],texts2[j],f, features))
				#ker[i][j] = approxkernel(texts1[i],texts2[j], f, features)
			else:
				poolDic[(i,j)] = pool.apply_async(kernel, args = (texts1[i],texts2[j],f))
				#ker[i][j] = kernel(texts1[i],texts2[j], f)
	pool.close()
	pool.join()
	ker = np.empty((len(texts2),len(texts1)))
	for key in poolDic:
		ker[key[1]][key[0]] = poolDic[key].get()
	f.cache_clear()
	return ker

def calculateSymMatrix(text, function, approx,features):
	pool = mp.Pool()
	poolDic = {}
	for i in range(0,len(text)):
		for j in range(0,i+1):
			if approx:
				poolDic[(i,j)] = pool.apply_async(approxkernel, args = (text[i],text[j],function, features))
				#ker[i][j] = ker[j][i] = approxkernel(text[i],text[j], function, features)
			else:
				poolDic[(i,j)] = pool.apply_async(kernel, args = (text[i],text[j],function))
				#ker[i][j] = ker[j][i] = kernel(text[i],text[j], function)
	pool.close()
	pool.join()
	ker = np.zeros((len(text),len(text)))
	for key in poolDic:
		ker[key[0]][key[1]] = ker[key[1]][key[0]] = poolDic[key].get()
	function.cache_clear()
	return ker

def getMatrix(train=True,func=ngram,approx=False):
	texts, targets = getTexts(True)
	features = []
	if(approx):
		features = featureSelection()
	if train:
		matrix = calculateSymMatrix(texts,func, approx,features)
	else:
		testTexts, targets = getTexts(train)
		matrix = calculateMatrix(texts,testTexts,func, approx,features)
	return matrix, targets

@functools.lru_cache(maxsize=None)
def getMatrixMemo(train=True,func=ssk,approx=False, k=3, lamda=0.5, label="corn"):
	origk = c.k
	origlamda = c.lamda
	origlabel = c.label

	c.k = k
	c.lamda = lamda
	c.label = label

	m,t =getMatrix(train=train,func=func,approx=approx)

	c.k = origk
	c.lamda = origlamda
	c.label = origlabel
	
	return m,t

def get2Matrix(train=True,func=ngram,approx=False, k1=2, k2=3,l1=0.01,l2=0.5):
	origk = c.k
	origlamda = c.lamda

	c.k = k1
	c.lamda = l1
	m1,t = getMatrix(train =train,func = func, approx=approx)

	c.k = k2
	c.lamda = l2
	m2,_ = getMatrix(train =train,func = func, approx=approx)

	c.k = origk
	c.lamda = origlamda
	return (m1+m2)/2,t

#print(getMatrixMemo())
#print(getMatrixMemo())

