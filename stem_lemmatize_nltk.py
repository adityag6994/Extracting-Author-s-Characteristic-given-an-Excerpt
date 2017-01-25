import numpy as np 
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
# import sklearn.feature_extraction.text 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

train_set = ("i am awesome  slightly on slight my bike, true truely truth bikes my biking schoolbag get off his bike and hand . A type of book.","In the summer of 1941 Grandma got sick and had to have an operation, so my birthday passed with little celebration. In the summer of 1940 we didn't do much for my birthday either, since the fighting had just ended in Holland. Grandma died in January 1942. No one knows how often I think of her and still love her. This birthday celebration in 1942 was intended to make up for the others, and Grandma's candle was lit along with the rest. The four of us are still doing well, and that brings me to the present date of June 20, 1942, and the solemn dedication of my diary.	SATURDAY, JUNE 20, 1942 Dearest Kitty! Let me get started right away; it's nice and quiet now. Father and Mother are out and Margot has gone to play Ping-Pong with some other young people at her friend Trees's. I've been playing a lot of Ping-Pong myself lately. So much that five of us girls have formed a club. It's called The Little Dipper Minus Two. A really silly name, but it's based on a mistake. We wanted to give our club a special name; and because there were five of us, we came up with the idea of the Little Dipper. We thought it consisted of five stars, but we turned out to be wrong. It has seven, like the Big Dipper, which explains the  Ilse Wagner has a Ping-Pong set, and the Wagners let us play in their big dining room whenever we want. Since we five Ping-Pong players like ice cream, especially in the summer, and since you")
# train_set = ("displayed a ghost tendency to towns seek qualities in displaying their next president that the current occupant of the Oval Office is perceived to lack. Contrast George W Bush's board-room manner with Bill Clinton's improvisational chaos or Jimmy Carter's squeaky clean reputation after Richard Nixon's Machiavellian manoeuvres.", "what is ghost towns up with you")
# test_set = ("Now displaying ghosts rocking tendencies they've met for the first time reputed ever - as president and president-elect. Talk about awkward.")#,"We can see the shining sun, the bright sun.")
test_set = ("I swerve slightly june summer on my bike, my bikes schoolbag falls, and the young man feels obliged to get off his bike and hand me the bag, by which time I've switched the conversation to another topic. These are the most innocent types. Of course, there are those who blow you kisses or try to take hold of your arm, but they're definitely knocking on the wrong door. I get off my bike and either refuse to make further use of their company or act as if I'm insulted and tell them in no uncertain terms to go on home without me. There you are. We've now laid the basis for our friendship. Until tomorrow.")
wnl = WordNetLemmatizer()
wpt = []; wpt_new = []
wpt_test = []
porter = PorterStemmer()
stopset = set(stopwords.words('english'))
# for itr in range(len(train_set)):
wpt=wordpunct_tokenize(train_set[0])
# for itr in range(len(train_set)): 
wpt_wo_stopwords = [w for w in wpt if not w in stopset]

# for itr in range(len(test_set)):
wpt_test=wordpunct_tokenize(test_set[0])
wpt_test = [w for w in wpt_test if not w in stopset]

# print wpt
data_stem = [porter.stem(t) for t in wpt]
data_lem= [wnl.lemmatize(t) for t in data_stem]
test_stem  = [porter.stem(t) for t in wpt_test]
test_lem = [wnl.lemmatize(t) for i in test_stem]

# count_vectorizer = TfidfVectorizer(stop_words='english')
# count_vectorizer.fit_transform(data_lem) # tokenizng
# # print "Vocabulary:", count_vectorizer.vocabulary_
# print len(count_vectorizer.vocabulary_)


# freq_term_matrix = count_vectorizer.transform(test_stem)
# freq_term_matrix1 = count_vectorizer1.transform(test_set)

# # print freq_term_matrix.todense()

# # TFIDF begin

# tfidf = TfidfTransformer(norm="l2")
# tfidf1 = TfidfTransformer(norm="l2")

# tfidf.fit(freq_term_matrix)
# tfidf1.fit(freq_term_matrix1)

# print freq_term_matrix
# print "IDF:", tfidf.idf_, len(tfidf.idf_) # freq in 
# print "IDF1:", tfidf1.idf_ , len(tfidf1.idf_)# freq in 

#### pos-tag
# print nltk.pos_tag(wpt)
# print
nltk_ptm = nltk.pos_tag(data_lem) 
merge = [x + y for x, y in nltk_ptm if y !='CD']
print nltk_ptm [0]
print len(merge), "merge"
print type(merge), type(train_set)
print train_set
merge = ' '.join(merge)
print eval(merge)
exit()
# count vectorize after pos taggin, removing 'CD' -- removes puncts 
count_vectorizer2 = TfidfVectorizer(stop_words='english')
dummy= count_vectorizer2.fit_transform(merge) # tokenizng
# print "Vocabulary:", count_vectorizer.vocabulary_
print (count_vectorizer2.vocabulary_)
print dummy
print type(tuple(merge)), type(train_set)

count_vectorizer1 = TfidfVectorizer(stop_words='english')
dummy1= count_vectorizer1.fit_transform(train_set)
# print "Vocabulary1:", count_vectorizer1.vocabulary_
print (count_vectorizer1.vocabulary_)
# print dummy1
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.pipeline import Pipeline
# categories = ['M', 'F']

# doc1 = "/home/praneeta/Downloads/bridget.txt"
# with open(doc1,'r') as f:
# 	data_doc1 = f.read().replace('\n','')
# # print data_doc1
# # exit()
# twenty_train = fetch_20newsgroups(subset='train',categories=categories, 
# 	shuffle=True, random_state=5)
# # text_file = open("Output.txt", "w")
# # text_file.write(twenty_train)
# # text_file.close()
# # print twenty_train
# # exit()

# twenty_train.target[:10]
# for t in twenty_train.target[:10]:
# 	 print(twenty_train.target_names[t])
# # twenty_train.target_names
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# # print X_train_counts
# tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
# X_train_tfidf = tfidf_transformer.transform(X_train_counts)
# #print X_train_tfidf

# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# docs_new = [data_doc1,"/home/praneeta/Downloads/temo/20news-bydate-test/F/Suck It, Wonder Woman!_ The Misadventures of a Hollywood Geek - Olivia Munn.txt","/home/praneeta/Downloads/temo/20news-bydate-test/F/Going Rogue_ An American Life - Sarah Palin.txt" ,"Bridget","married","Dexter","comic"]
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# # print X_new_tfidf

# predicted = clf.predict(X_new_tfidf)
# for doc, category in zip(docs_new, predicted):
# 	print(' => %s' % ( twenty_train.target_names[category]))

# text_clf = Pipeline([('vect', CountVectorizer()),
# 	('tfidf', TfidfTransformer()),
#  ('clf', MultinomialNB()),])

# text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
# docs_new = ['myths', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = text_clf.transform(X_new_counts)

# predicted = text_clf.predict(X_new_tfidf)
# for doc, category in zip(docs_new, predicted):
# 	print('%r => %s' % (doc, twenty_train.target_names[category]))

# Vocabulary: {u'oper': 64, u'summer': 90, u'dine': 25, u'love': 57, u'just': 48, u'celebr': 15, u'wagner': 102,
#  u'becaus': 7, u'rest': 74, u'bring': 12, u'dearest': 20, u'mistak': 61, u'make': 58, u'ha': 38, u'20': 3, u'1942': 
#  2, u'true': 95, u'1940': 0, u'1941': 1, u'father': 30, u'young': 106, u'littl': 55, u'fight': 31, u'late': 51, u'start': 89, 
#  u'dipper': 26, u'bike': 9, u'lot': 56, u'awesom': 5, u'type': 99, u'friend': 33, u'trueli': 96, u'diari': 22, u'form': 32,
#   u'especi': 28, u'club': 16, u'big': 8, u'hi': 40, u'know': 50, u'schoolbag': 78, u'kitti': 49, u'present': 71, u'like': 53,
#    u'dedic': 21, u'silli': 82, u'room': 76, u'consist': 17, u'whenev': 104, u'die': 24, u'didn': 23, u'tree': 94, u'truth': 97, 
#    u'januari': 46, u'candl': 14, u'gone': 35, u'seven': 80, u'set': 79, u've': 100, u'idea': 43, u'realli': 73, u'player': 69, 
#    u'right': 75, u'want': 103, u'pass': 65, u'intend': 45, u'girl': 34, u'saturday': 77, u'special': 87, u'cream': 18, u'end': 27, 
#    u'solemn': 86, u'away': 4, u'ping': 67, u'ice': 42, u'minu': 60, u'lit': 54, u'explain': 29, u'book': 11, u'sick': 81, u'peopl': 66, 
#    u'got': 36, u'nice': 63, u'birthday': 10, u'play': 68, u'star': 88, u'wa': 101, u'slight': 84, u'june': 47, u'hand': 39, u'wrong': 105, 
#    u'base': 6, u'let': 52, u'grandma': 37, u'date': 19, u'sinc': 83, u'came': 13, u'slightli': 85, u'quiet': 72, u'thought': 93, u'turn': 98, 
#    u'pong': 70, u'ils': 44, u'thi': 91, u'mother': 62, u'margot': 59, u'think': 92, u'holland': 41}
# 107
# Vocabulary1: {u'summer': 91, u'ilse': 45, u'love': 58, u'just': 49, u'girls': 37, u'wagner': 101, 
# u'rest': 76, u'ended': 30, u'dearest': 22, u'make': 59, u'little': 56, u'20': 3, u'1942': 2, u'1940': 0, 
# u'1941': 1, u'father': 33, u'young': 106, u'passed': 66, u'dipper': 28, u'bike': 8, u'stars': 89, u'set': 81, 
# u'playing': 71, u'type': 99, u'friend': 36, u'started': 90, u'lot': 57, u'holland': 42, u'club': 18, u'big': 7, 
# u'formed': 35, u'kitty': 50, u'trees': 94, u'truely': 96, u'diary': 24, u'true': 95, u'present': 73, u'like': 54, 
# u'room': 78, u'january': 47, u'didn': 25, u'bikes': 9, u'silly': 84, u'truth': 97, u'biking': 10, u'called': 14, 
# u'mistake': 62, u'gone': 38, u'seven': 82, u'right': 77, u've': 100, u'people': 67, u'doing': 29, u'awesome': 5, 
# u'turned': 98, u'idea': 44, u'schoolbag': 80, u'want': 103, u'candle': 16, u'operation': 65, u'saturday': 79, 
# u'special': 88, u'really': 75, u'date': 21, u'knows': 51, u'solemn': 87, u'away': 4, u'ping': 68, u'ice': 43, 
# u'lately': 52, u'lit': 55, u'dining': 27, u'book': 12, u'sick': 83, u'brings': 13, u'got': 39, u'cream': 20, 
# u'explains': 32, u'nice': 64, u'play': 69, u'intended': 46, u'slight': 85, u'june': 48, u'hand': 41, u'wrong': 105, 
# u'birthday': 11, u'let': 53, u'grandma': 40, u'dedication': 23, u'based': 6, u'wanted': 104, u'died': 26, u'celebration': 17, 
# u'especially': 31, u'slightly': 86, u'wagners': 102, u'quiet': 74, u'fighting': 34, u'minus': 61, u'thought': 93, 
# u'players': 70, u'pong': 72, u'mother': 63, u'margot': 60, u'think': 92, u'came': 15, u'consisted': 19}


# [('i', 'NN'), ('am', 'VBP'), ('awesome', 'RB'), ('slightly', 'RB'), ('on', 'IN'), ('slight', 'JJ'), ('my', 'PRP$'), ('bike', 'NN'), (',', ','), ('true', 'JJ'), ('truely', 'RB'), ('truth', 'NN'), ('bikes', 'NNS'), ('my', 'PRP$'), ('biking', 'NN'), ('schoolbag', 'NN'), ('get', 'VB'), ('off', 'RP'), ('his', 'PRP$'), ('bike', 'NN'), ('and', 'CC'), ('hand', 'NN'), ('.', '.'), ('A', 'DT'), ('type', 'NN'), ('of', 'IN'), ('book', 'NN'), ('.', '.'), ('In', 'IN'), ('the', 'DT'), ('summer', 'NN'), ('of', 'IN'), ('1941', 'CD'), ('Grandma', 'NNP'), ('got', 'VBD'), ('sick', 'JJ'), ('and', 'CC'), ('had', 'VBD'), ('to', 'TO'), ('have', 'VB'), ('an', 'DT'), ('operation', 'NN'), (',', ','), ('so', 'IN'), ('my', 'PRP$'), ('birthday', 'NN'), ('passed', 'VBD'), ('with', 'IN'), ('little', 'JJ'), ('celebration', 'NN'), ('.', '.'), ('In', 'IN'), ('the', 'DT'), ('summer', 'NN'), ('of', 'IN'), ('1940', 'CD'), ('we', 'PRP'), ('didn', 'VBP'), ("'", "''"), ('t', 'NN'), ('do', 'VBP'), ('much', 'RB'), ('for', 'IN'), ('my', 'PRP$'), ('birthday', 'NN'), ('either', 'RB'), (',', ','), ('since', 'IN'), ('the', 'DT'), ('fighting', 'NN'), ('had', 'VBD'), ('just', 'RB'), ('ended', 'VBN'), ('in', 'IN'), ('Holland', 'NNP'), ('.', '.'), ('Grandma', 'NNP'), ('died', 'VBD'), ('in', 'IN'), ('January', 'NNP'), ('1942', 'CD'), ('.', '.'), ('No', 'DT'), ('one', 'NN'), ('knows', 'VBZ'), ('how', 'WRB'), ('often', 'RB'), ('I', 'PRP'), ('think', 'VBP'), ('of', 'IN'), ('her', 'PRP$'), ('and', 'CC'), ('still', 'RB'), ('love', 'VB'), ('her', 'PRP'), ('.', '.'), ('This', 'DT'), ('birthday', 'JJ'), ('celebration', 'NN'), ('in', 'IN'), ('1942', 'CD'), ('was', 'VBD'), ('intended', 'VBN'), ('to', 'TO'), ('make', 'VB'), ('up', 'RP'), ('for', 'IN'), ('the', 'DT'), ('others', 'NNS'), (',', ','), ('and', 'CC'), ('Grandma', 'NNP'), ("'", 'POS'), ('s', 'NN'), ('candle', 'NN'), ('was', 'VBD'), ('lit', 'VBN'), ('along', 'IN'), ('with', 'IN'), ('the', 'DT'), ('rest', 'NN'), ('.', '.'), ('The', 'DT'), ('four', 'CD'), ('of', 'IN'), ('us', 'PRP'), ('are', 'VBP'), ('still', 'RB'), ('doing', 'VBG'), ('well', 'RB'), (',', ','), ('and', 'CC'), ('that', 'IN'), ('brings', 'VBZ'), ('me', 'PRP'), ('to', 'TO'), ('the', 'DT'), ('present', 'JJ'), ('date', 'NN'), ('of', 'IN'), ('June', 'NNP'), ('20', 'CD'), (',', ','), ('1942', 'CD'), (',', ','), ('and', 'CC'), ('the', 'DT'), ('solemn', 'JJ'), ('dedication', 'NN'), ('of', 'IN'), ('my', 'PRP$'), ('diary', 'JJ'), ('.', '.'), ('SATURDAY', 'NNP'), (',', ','), ('JUNE', 'NNP'), ('20', 'CD'), (',', ','), ('1942', 'CD'), ('Dearest', 'NNP'), ('Kitty', 'NNP'), ('!', '.'), ('Let', 'VB'), ('me', 'PRP'), ('get', 'VB'), ('started', 'VBN'), ('right', 'RB'), ('away', 'RB'), (';', ':'), ('it', 'PRP'), ("'", "''"), ('s', 'JJ'), ('nice', 'JJ'), ('and', 'CC'), ('quiet', 'JJ'), ('now', 'RB'), ('.', '.'), ('Father', 'NNP'), ('and', 'CC'), ('Mother', 'NNP'), ('are', 'VBP'), ('out', 'RB'), ('and', 'CC'), ('Margot', 'NNP'), ('has', 'VBZ'), ('gone', 'VBN'), ('to', 'TO'), ('play', 'VB'), ('Ping', 'VBG'), ('-', ':'), ('Pong', 'NN'), ('with', 'IN'), ('some', 'DT'), ('other', 'JJ'), ('young', 'JJ'), ('people', 'NNS'), ('at', 'IN'), ('her', 'PRP$'), ('friend', 'NN'), ('Trees', 'NNP'), ("'", 'POS'), ('s', 'NN'), ('.', '.'), ('I', 'PRP'), ("'", "''"), ('ve', 'RB'), ('been', 'VBN'), ('playing', 'VBG'), ('a', 'DT'), ('lot', 'NN'), ('of', 'IN'), ('Ping', 'VBG'), ('-', ':'), ('Pong', 'NNP'), ('myself', 'PRP'), ('lately', 'RB'), ('.', '.'), ('So', 'RB'), ('much', 'JJ'), ('that', 'IN'), ('five', 'CD'), ('of', 'IN'), ('us', 'PRP'), ('girls', 'NNS'), ('have', 'VBP'), ('formed', 'VBN'), ('a', 'DT'), ('club', 'NN'), ('.', '.'), ('It', 'PRP'), ("'", "''"), ('s', 'NN'), ('called', 'VBD'), ('The', 'DT'), ('Little', 'NNP'), ('Dipper', 'NNP'), ('Minus', 'NNP'), ('Two', 'CD'), ('.', '.'), ('A', 'DT'), ('really', 'RB'), ('silly', 'JJ'), ('name', 'NN'), (',', ','), ('but', 'CC'), ('it', 'PRP'), ("'", "''"), ('s', 'NNS'), ('based', 'VBN'), ('on', 'IN'), ('a', 'DT'), ('mistake', 'NN'), ('.', '.'), ('We', 'PRP'), ('wanted', 'VBD'), ('to', 'TO'), ('give', 'VB'), ('our', 'PRP$'), ('club', 'NN'), ('a', 'DT'), ('special', 'JJ'), ('name', 'NN'), (';', ':'), ('and', 'CC'), ('because', 'IN'), ('there', 'EX'), ('were', 'VBD'), ('five', 'CD'), ('of', 'IN'), ('us', 'PRP'), (',', ','), ('we', 'PRP'), ('came', 'VBD'), ('up', 'RP'), ('with', 'IN'), ('the', 'DT'), ('idea', 'NN'), ('of', 'IN'), ('the', 'DT'), ('Little', 'NNP'), ('Dipper', 'NNP'), ('.', '.'), ('We', 'PRP'), ('thought', 'VBD'), ('it', 'PRP'), ('consisted', 'VBD'), ('of', 'IN'), ('five', 'CD'), ('stars', 'NNS'), (',', ','), ('but', 'CC'), ('we', 'PRP'), ('turned', 'VBD'), ('out', 'RP'), ('to', 'TO'), ('be', 'VB'), ('wrong', 'JJ'), ('.', '.'), ('It', 'PRP'), ('has', 'VBZ'), ('seven', 'CD'), (',', ','), ('like', 'IN'), ('the', 'DT'), ('Big', 'NNP'), ('Dipper', 'NNP'), (',', ','), ('which', 'WDT'), ('explains', 'VBZ'), ('the', 'DT'), ('Ilse', 'NNP'), ('Wagner', 'NNP'), ('has', 'VBZ'), ('a', 'DT'), ('Ping', 'NNP'), ('-', ':'), ('Pong', 'NNP'), ('set', 'NN'), (',', ','), ('and', 'CC'), ('the', 'DT'), ('Wagners', 'NNP'), ('let', 'VBD'), ('us', 'PRP'), ('play', 'VB'), ('in', 'IN'), ('their', 'PRP$'), ('big', 'JJ'), ('dining', 'NN'), ('room', 'NN'), ('whenever', 'WRB'), ('we', 'PRP'), ('want', 'VBP'), ('.', '.'), ('Since', 'IN'), ('we', 'PRP'), ('five', 'CD'), ('Ping', 'VBG'), ('-', ':'), ('Pong', 'NN'), ('players', 'NNS'), ('like', 'IN'), ('ice', 'NN'), ('cream', 'NN'), (',', ','), ('especially', 'RB'), ('in', 'IN'), ('the', 'DT'), ('summer', 'NN'), (',', ','), ('and', 'CC'), ('since', 'IN'), ('you', 'PRP')]

# [(u'i', 'NN'), (u'am', 'VBP'), (u'awesom', 'RB'), (u'slightli', 'VBN'), (u'on', 'IN'), (u'slight', 'JJ'), (u'my', 'PRP$'), (u'bike', 'NN'), (u',', ','), (u'true', 'JJ'), (u'trueli', 'JJ'), (u'truth', 'NN'), (u'bike', 'IN'), (u'my', 'PRP$'), (u'bike', 'NN'), (u'schoolbag', 'NN'), (u'get', 'VB'), (u'off', 'RP'), (u'hi', 'NN'), (u'bike', 'NN'), (u'and', 'CC'), (u'hand', 'NN'), (u'.', '.'), (u'A', 'DT'), (u'type', 'NN'), (u'of', 'IN'), (u'book', 'NN'), (u'.', '.'), (u'In', 'IN'), (u'the', 'DT'), (u'summer', 'NN'), (u'of', 'IN'), (u'1941', 'CD'), (u'Grandma', 'NNP'), (u'got', 'VBD'), (u'sick', 'JJ'), (u'and', 'CC'), (u'had', 'VBD'), (u'to', 'TO'), (u'have', 'VB'), (u'an', 'DT'), (u'oper', 'NN'), (u',', ','), (u'so', 'IN'), (u'my', 'PRP$'), (u'birthday', 'NN'), (u'pas', 'NN'), (u'with', 'IN'), (u'littl', 'JJ'), (u'celebr', 'NN'), (u'.', '.'), (u'In', 'IN'), (u'the', 'DT'), (u'summer', 'NN'), (u'of', 'IN'), (u'1940', 'CD'), (u'we', 'PRP'), (u'didn', 'VBP'), (u"'", "''"), (u't', 'NN'), (u'do', 'VBP'), (u'much', 'RB'), (u'for', 'IN'), (u'my', 'PRP$'), (u'birthday', 'NN'), (u'either', 'CC'), (u',', ','), (u'sinc', 'VBD'), (u'the', 'DT'), (u'fight', 'NN'), (u'had', 'VBD'), (u'just', 'RB'), (u'end', 'VBN'), (u'in', 'IN'), (u'Holland', 'NNP'), (u'.', '.'), (u'Grandma', 'NNP'), (u'die', 'NN'), (u'in', 'IN'), (u'Januari', 'NNP'), (u'1942', 'CD'), (u'.', '.'), (u'No', 'DT'), (u'one', 'NN'), (u'know', 'VB'), (u'how', 'WRB'), (u'often', 'RB'), (u'I', 'PRP'), (u'think', 'VBP'), (u'of', 'IN'), (u'her', 'PRP$'), (u'and', 'CC'), (u'still', 'RB'), (u'love', 'VB'), (u'her', 'PRP'), (u'.', '.'), (u'Thi', 'NNP'), (u'birthday', 'JJ'), (u'celebr', 'NN'), (u'in', 'IN'), (u'1942', 'CD'), (u'wa', 'JJ'), (u'intend', 'VBP'), (u'to', 'TO'), (u'make', 'VB'), (u'up', 'RP'), (u'for', 'IN'), (u'the', 'DT'), (u'other', 'JJ'), (u',', ','), (u'and', 'CC'), (u'Grandma', 'NNP'), (u"'", 'POS'), (u's', 'NN'), (u'candl', 'NN'), (u'wa', 'NN'), (u'lit', 'VBD'), (u'along', 'RB'), (u'with', 'IN'), (u'the', 'DT'), (u'rest', 'NN'), (u'.', '.'), (u'The', 'DT'), (u'four', 'CD'), (u'of', 'IN'), (u'u', 'NN'), (u'are', 'VBP'), (u'still', 'RB'), (u'do', 'VBP'), (u'well', 'RB'), (u',', ','), (u'and', 'CC'), (u'that', 'IN'), (u'bring', 'VBG'), (u'me', 'PRP'), (u'to', 'TO'), (u'the', 'DT'), (u'present', 'JJ'), (u'date', 'NN'), (u'of', 'IN'), (u'June', 'NNP'), (u'20', 'CD'), (u',', ','), (u'1942', 'CD'), (u',', ','), (u'and', 'CC'), (u'the', 'DT'), (u'solemn', 'JJ'), (u'dedic', 'NN'), (u'of', 'IN'), (u'my', 'PRP$'), (u'diari', 'NN'), (u'.', '.'), (u'SATURDAY', 'NNP'), (u',', ','), (u'JUNE', 'NNP'), (u'20', 'CD'), (u',', ','), (u'1942', 'CD'), (u'Dearest', 'NNP'), (u'Kitti', 'NNP'), (u'!', '.'), (u'Let', 'VB'), (u'me', 'PRP'), (u'get', 'VB'), (u'start', 'RB'), (u'right', 'JJ'), (u'away', 'RB'), (u';', ':'), (u'it', 'PRP'), (u"'", "''"), (u's', 'JJ'), (u'nice', 'JJ'), (u'and', 'CC'), (u'quiet', 'JJ'), (u'now', 'RB'), (u'.', '.'), (u'Father', 'NNP'), (u'and', 'CC'), (u'Mother', 'NNP'), (u'are', 'VBP'), (u'out', 'RB'), (u'and', 'CC'), (u'Margot', 'NNP'), (u'ha', 'VBP'), (u'gone', 'VBN'), (u'to', 'TO'), (u'play', 'VB'), (u'Ping', 'VBG'), (u'-', ':'), (u'Pong', 'NN'), (u'with', 'IN'), (u'some', 'DT'), (u'other', 'JJ'), (u'young', 'JJ'), (u'peopl', 'NN'), (u'at', 'IN'), (u'her', 'PRP$'), (u'friend', 'NN'), (u'Tree', 'NNP'), (u"'", 'POS'), (u's', 'NN'), (u'.', '.'), (u'I', 'PRP'), (u"'", "''"), (u've', 'NN'), (u'been', 'VBN'), (u'play', 'VBP'), (u'a', 'DT'), (u'lot', 'NN'), (u'of', 'IN'), (u'Ping', 'VBG'), (u'-', ':'), (u'Pong', 'NNP'), (u'myself', 'PRP'), (u'late', 'RB'), (u'.', '.'), (u'So', 'RB'), (u'much', 'JJ'), (u'that', 'IN'), (u'five', 'CD'), (u'of', 'IN'), (u'u', 'JJ'), (u'girl', 'NNS'), (u'have', 'VBP'), (u'form', 'VBN'), (u'a', 'DT'), (u'club', 'NN'), (u'.', '.'), (u'It', 'PRP'), (u"'", "''"), (u's', 'JJ'), (u'call', 'VB'), (u'The', 'DT'), (u'Littl', 'NNP'), (u'Dipper', 'NNP'), (u'Minu', 'NNP'), (u'Two', 'CD'), (u'.', '.'), (u'A', 'DT'), (u'realli', 'JJ'), (u'silli', 'NN'), (u'name', 'NN'), (u',', ','), (u'but', 'CC'), (u'it', 'PRP'), (u"'", "''"), (u's', 'JJ'), (u'base', 'NN'), (u'on', 'IN'), (u'a', 'DT'), (u'mistak', 'NN'), (u'.', '.'), (u'We', 'PRP'), (u'want', 'VBP'), (u'to', 'TO'), (u'give', 'VB'), (u'our', 'PRP$'), (u'club', 'NN'), (u'a', 'DT'), (u'special', 'JJ'), (u'name', 'NN'), (u';', ':'), (u'and', 'CC'), (u'becaus', 'VB'), (u'there', 'EX'), (u'were', 'VBD'), (u'five', 'CD'), (u'of', 'IN'), (u'u', 'NN'), (u',', ','), (u'we', 'PRP'), (u'came', 'VBD'), (u'up', 'RP'), (u'with', 'IN'), (u'the', 'DT'), (u'idea', 'NN'), (u'of', 'IN'), (u'the', 'DT'), (u'Littl', 'NNP'), (u'Dipper', 'NNP'), (u'.', '.'), (u'We', 'PRP'), (u'thought', 'VBD'), (u'it', 'PRP'), (u'consist', 'NN'), (u'of', 'IN'), (u'five', 'CD'), (u'star', 'NN'), (u',', ','), (u'but', 'CC'), (u'we', 'PRP'), (u'turn', 'VBP'), (u'out', 'RP'), (u'to', 'TO'), (u'be', 'VB'), (u'wrong', 'JJ'), (u'.', '.'), (u'It', 'PRP'), (u'ha', 'VBD'), (u'seven', 'CD'), (u',', ','), (u'like', 'IN'), (u'the', 'DT'), (u'Big', 'NNP'), (u'Dipper', 'NNP'), (u',', ','), (u'which', 'WDT'), (u'explain', 'VBP'), (u'the', 'DT'), (u'Ils', 'NNP'), (u'Wagner', 'NNP'), (u'ha', 'VBD'), (u'a', 'DT'), (u'Ping', 'NNP'), (u'-', ':'), (u'Pong', 'NNP'), (u'set', 'NN'), (u',', ','), (u'and', 'CC'), (u'the', 'DT'), (u'Wagner', 'NNP'), (u'let', 'NN'), (u'u', 'JJ'), (u'play', 'VB'), (u'in', 'IN'), (u'their', 'PRP$'), (u'big', 'JJ'), (u'dine', 'NN'), (u'room', 'NN'), (u'whenev', 'IN'), (u'we', 'PRP'), (u'want', 'VBP'), (u'.', '.'), (u'Sinc', 'NNP'), (u'we', 'PRP'), (u'five', 'CD'), (u'Ping', 'VBG'), (u'-', ':'), (u'Pong', 'NNP'), (u'player', 'NN'), (u'like', 'IN'), (u'ice', 'NN'), (u'cream', 'NN'), (u',', ','), (u'especi', 'NN'), (u'in', 'IN'), (u'the', 'DT'), (u'summer', 'NN'), (u',', ','), (u'and', 'CC'), (u'sinc', 'NN'), (u'you', 'PRP')]