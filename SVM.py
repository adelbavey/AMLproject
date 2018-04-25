import codecs
from sklearn import svm, datasets
from sklearn.metrics import f1_score
from numpy.linalg import norm
import numpy as np
import time
import constants as c
import kernels

def testAlignment(f):
    time1 = time.time()
    #print("Get true training matrix")
    gram, targets = kernels.getMatrixMemo(train=True,func=f,approx=False,k=c.k,lamda=c.lamda, label=c.label)
    #print(c.k,c.lamda,c.label, c.nFeatures)
    #print("Train true kernel")
    clf = svm.SVC(kernel="precomputed")
    clf.fit(gram,targets)
    time2 = time.time()
    #print("Testing true kernel")
    testGram, testTargets = kernels.getMatrixMemo(train=False,func=f,approx=False, k=c.k,lamda=c.lamda, label=c.label)
    predictedTrue = clf.predict(testGram)
    time3 = time.time()
    #print("Get Approx training matrix")
    appGram, targets = kernels.getMatrix(train=True,func=f,approx=True)
    #print("Train approx kernel")
    clf = svm.SVC(kernel="precomputed")
    clf.fit(appGram,targets)
    time4 = time.time()
    #print("Testing approx kernel")
    testGram, testTargets = kernels.getMatrix(train=False,func=f,approx=True)
    predictedApprox = clf.predict(testGram)
    time5 = time.time()
    
    #print("gram:" + str(gram))
    #print("appgram:" + str(appGram))
    print("Real kernel training took:   " + str((time2-time1)) + " seconds")
    print("Real kernel total took:      " + str((time3-time1)) + " seconds")
    print("Approximating training took: " + str((time4-time3)) + " seconds")
    print("Approximating total took:    " + str((time5-time3)) + " seconds")
    print("Alignment: " + str(alignment(gram,appGram)))

    print("F1 True: " + str(f1_score(testTargets,predictedTrue)))
    print("Precision True: " + str(precision(testTargets,predictedTrue)))
    print("recall True: " + str(recall(testTargets,predictedTrue)))
    print("accuracy True: " + str(accuracy(testTargets,predictedTrue)))

    print("F1 Approx: " + str(f1_score(testTargets,predictedApprox)))
    print("Precision Approx: " + str(precision(testTargets,predictedApprox)))
    print("recall Approx: " + str(recall(testTargets,predictedApprox)))
    print("accuracy Approx: " + str(accuracy(testTargets,predictedApprox)))


def alignment(K1,K2):
    return frobProduct(K1,K2)/((frobProduct(K1,K1)*frobProduct(K2,K2))**(1.0/2))

def frobProduct(K1, K2):
    return sum([sum([K1[i][j]*K2[i][j] for i in range(0,K1.shape[0])]) for j in range(0,K2.shape[1])])

def testKernel(f,approx=False):
    time1 = time.time()
    #print("Get training matrix")
    gram, targets = kernels.getMatrix(train=True,func=f,approx=approx)
    clf = svm.SVC(kernel="precomputed")
    #print("Training")
    clf.fit(gram,targets)
    time2 = time.time()
    #print("Testing")
    testGram, testTargets = kernels.getMatrix(train=False,func=f,approx=approx)
    predicted = clf.predict(testGram)
    time3 = time.time()
    #print("train targets")
    #print(targets)
    #print("True Targets")
    #print(testTargets)
    #print("predicted targets")
    #print(predicted)
    print("F1 : " + str(f1_score(testTargets,predicted)))
    print("Precision : " + str(precision(testTargets,predicted)))
    print("recall : " + str(recall(testTargets,predicted)))
    print("accuracy : " + str(accuracy(testTargets,predicted)))
    print("Training Time: " + str(time2-time1))
    print("Total Time: " + str(time3-time1))

def testDoubleKernel(f,approx=False,k1=2,k2=3,l1=0.5,l2=0.6):
    time1 = time.time()
    #print("Get training matrix")
    gram, targets = kernels.get2Matrix(train=True,func=f,approx=approx,k1=k1,k2=k2,l1=l1,l2=l2)
    clf = svm.SVC(kernel="precomputed")
    #print("Training")
    clf.fit(gram,targets)
    time2 = time.time()
    #print("Testing")
    testGram, testTargets = kernels.get2Matrix(train=False,func=f,approx=approx,k1=k1,k2=k2,l1=l1,l2=l2)
    predicted = clf.predict(testGram)
    time3 = time.time()
    #print("train targets")
    #print(targets)
    #print("True Targets")
    #print(testTargets)
    #print("predicted targets")
    #print(predicted)
    print("F1 : " + str(f1_score(testTargets,predicted)))
    print("Precision : " + str(precision(testTargets,predicted)))
    print("recall : " + str(recall(testTargets,predicted)))
    print("accuracy : " + str(accuracy(testTargets,predicted)))
    print("Training Time: " + str(time2-time1))
    print("Total Time: " + str(time3-time1))

def accuracy(targets,predTargets):
    corrects = 0.0
    for i in range(0,len(predTargets)):
        if predTargets[i] == targets[i]:
            corrects +=1.0
    return corrects/len(predTargets)

def precision(targets, predTargets):
    falsePos = 0.0
    trutru = 0.0
    for i in range(0,len(targets)):
        if targets[i] == predTargets[i] and targets[i] == 1:
            trutru += 1
        elif(predTargets[i] == 1):
            falsePos+=1
    if (trutru+falsePos) == 0:
        return 0
    return trutru/(trutru+falsePos)

def recall(targets,predTargets):
    total = 0.0
    trutru = 0.0
    for i in range(0,len(targets)):
        if targets[i] == 1:
            total+=1
            if targets[i]==predTargets[i]:
                trutru+=1
    if total==0:
        return 0
    return trutru/total

def f1_score(targets,predTargets):
    p = precision(targets,predTargets)
    r = recall(targets,predTargets)
    if (p+r) == 0:
        return 0
    return 2*(p*r)/(p+r)

#testAlignment(kernels.ngram)
#testKernel(kernels.bag)
#testAlignment(kernels.ssk)
#print(recall([1,1,0,0],[1,1,1,1]))
#50 samples,k=5, 0 score, train 1640 sek,total 6028
#F1 : 0 Precision : 0.0 recall : 0.0 Training Time: 10359.741450548172 Total Time: 20970.61513018608

