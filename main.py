from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.texwt import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn import linear_model

#from .base import load_files
import csv

PATH_HOME='/home/namrita/Downloads/AIdata/'
FILESNAME_MASTER='lookuptable.csv'
PATH_TEST='test/'
PATH_TRAIN='train/'

#Parser handler (reading user input for various options)

op = OptionParser()
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
(opts, args) = op.parse_args()

dat = csv.reader(open(PATH_HOME+FILESNAME_MASTER, 'r'), delimiter = ';')

labels = ([h[0] for h in dat][0]).split(',')
columns = []
with open(PATH_HOME+FILESNAME_MASTER,'r') as f:
	reader = csv.DictReader(f)
	g= [r for r in reader]
	f.close()
	for j in range(len(labels)):
		for i in range(len(g)):
			columns.append([v for k,v in g[i].iteritems() if k==labels[j]])
f.close()
columns= np.array(columns).reshape(len(labels),len(g))
no_labels, no_samples = np.shape(columns)

dict_labels = {labels[itr]:columns[itr] for itr in range(no_labels)}
filenames = dict_labels[labels[0]]
# print(labels[0])
filepaths= [PATH_HOME+PATH_TRAIN+'J/'+x for x in dict_labels[labels[0]]]

#Reading files from database and storing them according to our need
def loadtrain(label):
	k=load_files(PATH_HOME+PATH_TRAIN, encoding='latin1')
	filename_with_data = {f:d for f,d in zip(k.filenames, k.data)}
	data_o=[filename_with_data.get(i) for i in np.array(filepaths) if i in k.filenames]
	k.filenames=np.array([x for x in filepaths if x in k.filenames])
	k.data=data_o
	y = search_y(label,k.filenames,PATH_TRAIN+'J/')
	k.target=y
	return k

#getting labels
def get_y(label):
	y=list()
	if label=='Gender':
		z=dict_labels['Gender']
		l=len(z)
		for i in range(0,l):
			if z[i]=='M':
				y.append(1.0)
			else:
				y.append(0.0)

	if label=='Genre1':
		z=dict_labels['Genre']
		l=len(z)
		z1=['F','R','FA','CF','HF','SF','HD','T','PF','P','PD','H']
		z2=['AB','B','NF']
		z3=['G','A']
		z4=['C']
		for i in range(0,l):
			if z[i] in z1:
				y.append(0.0)
			if z[i] in z2:
				y.append(1.0)
			if z[i] in z3:
				y.append(2.0)
			else:
				y.append(3.0)
	if label=='Genre2':
		z=dict_labels['Genre']
		l=len(z)
		z1=['AB','B']
		z2=['NF']
		z3=['F','R','HD','PF']
		z4=['H']
		z5=['FA']
		z6=['CF','HF','T']
		z7=['P','PD']
		z8=['G','A']
		z9=['C']
		z10=['SF']
		for i in range(0,l):
			if z[i] in z1:
				y.append(0.0)
			if z[i] in z2:
				y.append(1.0)
			if z[i] in z3:
				y.append(2.0)
			if z[i] in z4:
				y.append(3.0)
			if z[i] in z5:
				y.append(4.0)
			if z[i] in z6:
				y.append(5.0)
			if z[i] in z7:
				y.append(6.0)
			if z[i] in z8:
				y.append(7.0)
			if z[i] in z9:
				y.append(8.0)
			else:
				y.append(9.0)

	if label=='yob1':
		z=dict_labels['YOB']
		l=len(z)
		for i in range(0,l):
			if z[i]>1950:
				y.append(2.0)
			if z[i]<1900:
				y.append(0.0)
			else:
				y.append(1.0)

	if label=='yob2':
		z=dict_labels['YOB']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])

	if label=='yop':
		z=dict_labels['YOP']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])			

	if label=='Age':
		z=dict_labels['Age']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])

	return y

#feature extraction
def feature_extraction(label,train_data):
	# train_data=loadtrain(label)
	if opts.use_hashing:
	    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
	                                   n_features=opts.n_features)
	    X_train = vectorizer.transform(train_data.data)
	    feature_names = None

	else:
	    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
	                                 stop_words='english')
	    X_train = vectorizer.fit_transform(train_data.data)
	    # mapping from integer feature name to original token string
	    feature_names = vectorizer.get_feature_names()
        feature_names = np.asarray(feature_names)


	ch2=None
	if opts.select_chi2:
	    print("Extracting %d best features by a chi-squared test" %
	          opts.select_chi2)
	    t0 = time()
	    ch2 = SelectKBest(chi2, k=opts.select_chi2)
	    X_train = ch2.fit_transform(X_train, y_train)
	    X_test = ch2.transform(X_test)
	    if feature_names:
	        # keep selected feature names
	        feature_names = [feature_names[i] for i
	                         in ch2.get_support(indices=True)]
	return X_train, feature_names, ch2, vectorizer   

#loading test files
def load_test_files(transformer):
	t=load_files(PATH_HOME+PATH_TEST,encoding='latin1')
	feature_vector = transformer.transform(t.data)

	return t,feature_vector


#search for features
def search_y(label,namelist,mid_path):
	fp = [PATH_HOME+mid_path+x for x in filenames]

	filename_with_label = {f:d for f,d in zip(fp,get_y(label))}#dict_labels('filepaths'), dict_labels(label))}
	# print(filename_with_label)
	k=[filename_with_label[i] for i in namelist]
	return k

#functions for various classifier
def simple_classify(clf,test_x,test_y,train_x,train_y):

    t0 = time()
    clf.fit(train_x,train_y)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_x)

    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(test_y, pred)
    print("accuracy:   %0.3f" % score)
    return zip(pred,test_y)

def bayesian_ridge_regression(test_x,test_y,train_y,train_x):
	reg = linear_model.BayesianRidge()
	train_x=train_x.toarray().astype(np.float)
	# print(type(train_x))
	train_y=np.array(train_y).astype(np.float)
	# print(type(k.target))
	t0 = time()	
	print(reg.fit(train_x,train_y))
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
  	t0 = time()	
	pred=reg.predict (test_x)
	test_time = time() - t0
 	print("test time:  %0.3fs" % test_time)
 	print(zip(pred,test_y))
	test_y = [float(x) for x in test_y]
	count=0;
	l=len(test_y)
	for i in range(0,l):  
		if (abs(pred[i]-test_y[i])<10):
			count=count+1
	count=float(count)
	l=float(l)
	score=(count/l)
	print("Score",score)


def ridge_regression(test_x,test_y,train_y,train_x):
	reg = linear_model.Ridge (alpha = .5)
	train_x=train_x.toarray().astype(np.float)
	# print(type(train_x))
	train_y=np.array(train_y).astype(np.float)
	t0 = time()
	# print(type(k.target))
	reg.fit(train_x,train_y)
	train_time = time() - t0
 	print("train time: %0.3fs" % train_time)
  	t0 = time()
	pred=reg.predict (test_x)
	test_time = time() - t0
 	print("test time:  %0.3fs" % test_time)
 	print(zip(pred,test_y))
	test_y = [float(x) for x in test_y]
	count=0;
	l=len(test_y)
	for i in range(0,l):  
		if (abs(pred[i]-test_y[i])<10):
			count=count+1
	count=float(count)
	l=float(l)
	score=(count/l)
	print("Score",score)
	fig, ax = plt.subplots()
	ax.scatter(test_y, pred)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

#main code to use all the functions

givenlabel='Age'
k=loadtrain(givenlabel)
# print (k)
train_x,f_names,chi,transformer=feature_extraction(givenlabel,k)
test,test_x=load_test_files(transformer)
test_y=search_y(givenlabel,test.filenames,PATH_TEST+'J/')

reg_labels=['yob2','yop','Age']
if givenlabel in reg_labels:
	bayesian_ridge_regression(test_x,test_y,k.target,train_x)
	# ridge_regression(test_x,test_y,k.target,train_x)
	
	
else:
    print(simple_classify(MultinomialNB(alpha=.01),test_x,test_y,train_x,k.target))