from sklearn.feature_extraction.text import CountVectorizer
train_set = ("Premchand was born on 31 July 1880 in Lamhi, a village located near Varanasi",
		" (Banaras). His ancestors came from a large Kayastha family, which owned six",
		" bighas of land. His grandfather Guru Sahai Rai was a patwari (village",
		" land record-keeper), and his father Ajaib Rai was a post office clerk.")
test_set = ("My name is Premchand", 
    "I too am from Lamhi")

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer.fit_transform(train_set)

print count_vectorizer.vocabulary_

#{u'blue': 0, u'sun': 3, u'bright': 1, u'sky': 2}

freq_term_matrix = count_vectorizer.transform(test_set)

#print freq_term_matrix.todense()
#print smatrix
#smatrix.todense()
#print smatrix.todense()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

#print "IDF:", tfidf.idf_
#IDF: [ 2.09861229  1.          1.40546511  1.        ]

tf_idf_matrix = tfidf.transform(freq_term_matrix)
print tf_idf_matrix.todense()

#final_out_put
#[[ 0.          0.50154891  0.70490949  0.50154891]
#[ 0.          0.4472136   0.          0.89442719]]


