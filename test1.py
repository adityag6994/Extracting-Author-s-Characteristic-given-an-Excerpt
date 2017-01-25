# import numpy as np 
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
# # import sklearn.feature_extraction.text 
# from sklearn.feature_extraction.text import TfidfTransformer
# # from sklearn.datasets import fetch_20newsgroups

# train_set = ("The sky is blue.", "The sun is bright.")
# test_set = ("The sun in the sky is bright.",
# "We can see the shining sun, the bright sun.")

# count_vectorizer = CountVectorizer(stop_words='english')
# count_vectorizer.fit_transform(train_set) # tokenizng
# print "Vocabulary:", count_vectorizer.vocabulary_

# # count_vectorizer1 = CountVectorizer()
# # count_vectorizer1.fit_transform(train_set)
# # print "Vocabulary1:", count_vectorizer1.vocabulary_
# # Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}

# freq_term_matrix = count_vectorizer.transform(test_set)
# print freq_term_matrix.todense()

# # TFIDF begin

# tfidf = TfidfTransformer(norm="l2")
# tfidf.fit(freq_term_matrix)
# print freq_term_matrix
# print "IDF:", tfidf.idf_ # freq in 

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian',
'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories, 
	shuffle=True, random_state=42)
twenty_train.target[:10]
for t in twenty_train.target[:10]:
	 print(twenty_train.target_names[t])
# twenty_train.target_names
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print X_train_counts
tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
# print X_train_tf

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['myths', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, twenty_train.target_names[category]))
