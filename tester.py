import SVM
import constants as c
import kernels
import sys

#todo

ks = [3,4,5,6,7]
lamdas=[0.01,0.05,0.09,0.3,0.7]
topics=["corn","acq","earn","crude"]
kers = [kernels.bag, kernels.ngram, kernels.ssk]

def baseKSSKtest():
	c.trainSamples = 80
	c.testSamples = 20
	c.lamda = 0.05
	for topic in topics:
		c.label = topic
		print("-------------")
		print("SSK " + topic)
		print("-------------")
		for k in ks:
			c.k = k
			print(topic + " SSK " + str(c.k), file=sys.stderr)
			print("K = " + str(c.k))
			SVM.testKernel(kernels.ssk)
			print("-------------")
		print("-------------")

def baseKNgramBagtest():
	c.trainSamples = 80
	c.testSamples = 20
	c.lamda = 0.05
	for topic in topics:
		c.label = topic
		print("-------------")
		print("Ngram " + topic)
		print("-------------")
		for k in ks:
			c.k = k
			print(topic + " Ngram " + str(c.k), file=sys.stderr)	
			print("K = " + str(c.k))
			SVM.testKernel(kernels.ngram)
			print("-------------")
		print("-------------")

		print("BagOW " + topic)
		print("-------------")
		print(topic + " BagOW", file=sys.stderr)
		SVM.testKernel(kernels.bag)
		print("-------------")


def baseLamdatest():
	c.trainSamples = 80
	c.testSamples = 20
	c.k = 5
	for topic in topics:
		c.label = topic
		print("-------------")
		print("SSK " + topic)
		print("-------------")
		for l in lamdas:
			c.lamda = l
			print(topic + " SSK " + str(c.lamda), file=sys.stderr)
			print("Lamda = " + str(c.lamda))
			SVM.testKernel(kernels.ssk)
			print("-------------")
		print("-------------")

def baseCombinedKtest():
	c.trainSamples = 80
	c.testSamples = 20
	c.lamda = 0.05
	K = [(3,4),(4,5),(5,6)]
	for topic in topics:
		c.label = topic
		print("-------------")
		print("CombinedSSK for K " + topic)
		print("-------------")
		for k in K:
			print(topic + " CombinedSSK for K " + str(k), file=sys.stderr)
			print("K = " + str(k))
			SVM.testDoubleKernel(kernels.ssk,k1=k[0],k2=k[1],l1=c.lamda,l2=c.lamda)
			print("-------------")
		print("-------------")

def baseCombinedLamdatest():
	c.trainSamples = 80
	c.testSamples = 20
	c.k = 5
	for topic in topics:
		c.label = topic
		print("-------------")
		print("CombinedSSK for Lamda " + topic)
		print("-------------")
		for lamda in [(0.05,0.1),(0.1,0.3),(0.3,0.5)]:
			print(topic + " CombinedSSK for Lamda " + str(lamda), file=sys.stderr)
			print("Lamda = " + str(lamda))
			SVM.testDoubleKernel(kernels.ssk,k1=c.k,k2=c.k,l1=lamda[0],l2=lamda[1])
			print("-------------")
		print("-------------")

def approxSSKTest():
	c.trainSamples = 80
	c.testSamples = 20
	c.lamda = 0.05
	for topic in topics:
		c.label = topic
		print("-------------")
		print("ApproxSSK " + topic)
		print("-------------")
		for nF in [100, 700, 1800]:
			c.nFeatures = nF
			for k in [3,4,5]:
				c.k = k
				c.nFeatures = nF
				print(topic + " ApproxSSK NumberOfFeatures = " + str(c.nFeatures) + " K = " + str(c.k), file=sys.stderr)
				print ("K = " + str(c.k) + " lamda = " + str(c.lamda) + " NumberOfFeatures = " + str(c.nFeatures))
				SVM.testAlignment(kernels.ssk)
				print("-------------")
			print("-------------")
		print("-------------")

def approxSSKNgramWordTest():
	c.trainSamples = 380
	c.testSamples = 90
	c.lamda = 0.05
	c.nFeatures = 100
	ks = [4,5,6]
	for topic in topics:
		c.label = topic
		print("-------------")
		print("ApproxSSK " + topic)
		print("-------------")
		for k in ks:
			c.k = k
			print(topic + " ApproxSSK " + str(c.k), file=sys.stderr)
			print("K = " + str(c.k))
			SVM.testKernel(kernels.ssk,True)
			print("-------------")
		print("-------------")

		print("Ngram " + topic)
		print("-------------")
		for k in ks:
			c.k = k
			print(topic + " Ngram " + str(c.k), file=sys.stderr)	
			print("K = " + str(c.k))
			SVM.testKernel(kernels.ngram)
			print("-------------")
		print("-------------")

		print("BagOW " + topic)
		print("-------------")
		print(topic + " BagOW", file=sys.stderr)
		SVM.testKernel(kernels.bag)
		print("-------------")

if(len(sys.argv)==1):
	pass
	#baseKtest()
	#baseLamdatest()
	#baseCombinedKtest()
	#baseCombinedLamdatest()
	#approxSSKTest()
elif(sys.argv[1]=="kssk"):
	baseKSSKtest()
elif(sys.argv[1]=="kngrambag"):
	baseKNgramBagtest()
elif(sys.argv[1]=="l"):
	baseLamdatest()
elif(sys.argv[1]=="ck"):
	baseCombinedKtest()
elif(sys.argv[1]=="cl"):
	baseCombinedLamdatest()
elif(sys.argv[1]=="approx"):
	approxSSKTest()
elif(sys.argv[1]=="approxcompare"):
	approxSSKNgramWordTest()
