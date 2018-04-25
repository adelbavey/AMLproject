import os
import re, string
import codecs
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

cachedStopWords = stopwords.words("english")
punctuations = [",",".","?","!",";",":","(",")","[","]","-","\"", "\\", "/", "<", ">", "\'"]
path = "./reuters21578/"
counter = 0
whitelist = ["earn","crude","corn","acq"]

def checkTopic(whitelist, topics):
	for topic in topics.find_all("d"):
		for white in whitelist:
			if white in topic:
				return True
	return False

def parseFile(filename):
	global counter
	with codecs.open(path + filename,"r", encoding="utf-8", errors="replace") as testfile:
		soup = BeautifulSoup(testfile.read(), 'html.parser')
		for thing in soup.find_all("reuters"):
			if thing["topics"] == "YES" and thing.topics.d and thing.body and thing["lewissplit"] == "TEST":
				if checkTopic(whitelist, thing.topics):
					counter += 1
					print("<reuters>")
					print(thing.topics)
					noStopWords = ' '.join([word.lower() for word in thing.body.text.split() if word not in cachedStopWords])
					noStopWords = ''.join([word for word in noStopWords if word not in punctuations])
					noStopWords = noStopWords[:-9]
					noStopWords = re.sub('[%s]' %string.digits, ' ', noStopWords)
					body = ""
					for char in noStopWords:
						if not(ord(char) > ord('z') or ord(char) < ord('a')) or re.match('\s',char):
							body += char
					print("<body>"+body+"</body>")
					print("</reuters>")


for filename in os.listdir(path):
	if "sgm" in filename:
		parseFile(filename)
