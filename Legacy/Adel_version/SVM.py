import preproccess, codecs
from sklearn import svm, datasets
import numpy as np
import kernel
import preproccess

label = "acq"

#For testing
stories = []
sap1 = {}
sap2 = {}
#ssk takes in two stories(strings) and returns the kernel value
def ssk():
    pass

'''
def ssk(X,Y):
    #print(np.dot(X,X.T).shape)
    #print(np.dot(Y,Y.T).shape)
    #print((np.dot(np.dot(X,X.T),np.dot(Y,Y.T)).shape))
    #left = X/(sum([x**2 for x in X])**(1.0/2))
    left = [x/((sum([i**2 for i in x])**(1.0/2))) for x in X]
    #right = Y/(sum([y**2 for y in Y])**(1.0/2))
    right = [y/((sum([i**2 for i in y])**(1.0/2))) for y in Y]
    return np.dot(np.array(left),np.array(right).T)
    #np.dot(X,Y.T)/((np.dot(np.dot(X,X.T),np.dot(Y,Y.T)))**(1.0/2))
'''
def getInputData(label,samples, path="train.txt"):
    return preproccess.generateFeatureVector(label,samples,path)

def accuracy(clf,data,target):
    predictions = clf.predict(data)
    corrects = 0.0
    for i in range(0,len(predictions)):
        if predictions[i] == target[i]:
            corrects +=1.0
    accuracy = corrects/len(predictions)
    return accuracy

def test(clf, testLabel, testSamples=8):
    global sap2,stories,sap1
    teststories, sap2,targets = preproccess.getData(preproccess.targetClass, testSamples, path="test.txt")
    gram = kernel.constructGram(teststories,stories, preproccess.k,sap2,sap1)

    print(accuracy(clf, gram, targets))

def trainSVM(label,samples=8):
    global stories, sap1
    clf = svm.SVC(kernel='precomputed')
    stories, sap1, targets = preproccess.getData(preproccess.targetClass, samples)
    gram = kernel.constructGram(stories,stories, preproccess.k, sap1, sap1)
    clf.fit(gram,targets)
    print accuracy(clf,gram,targets)
    return clf

cornclf = trainSVM(label)
test(cornclf, label)
