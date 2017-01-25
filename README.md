# author-profiling
This paper presents a method to predict the characteristic traits of an author of a given book. Characteristic traits - like

gender, race, age of the author at the time of publishing, the genre of the book, and other socioeconomic and

psychological factors - which influence the content and style of writing of an author can be predicted given his/her

work. Multiple feature vectors(tf-idf, n-gram, POS tagging) were extracted from texts using preprocessing techniques

(stopword removal, stemming, lemmatization) and a comparative performance study for every technique was analyzed

by applying a six classification methods (namely Naïve Bayes, SVM using gradient descent, k-NN, Random Forest,

Passive Aggressive, Perceptron). These analytic studies were performed with the aid of the sklearn machine learning

library and the Natural Language Toolkit. The performance of the classifiers was further compared for data with and

without preprocessing.
