from urllib import urlopen
from bs4 import BeautifulSoup
import numpy as np
webpage = urlopen("http://www.biography.com/.amp/people/sylvia-plath-9442550").read()
soup = BeautifulSoup(webpage, "html5lib")
# soup = BeautifulSoup(open("http://www.biography.com/people/sylvia-plath-9442550"))
x = soup.findAll('script', type="application/ld+json")
x = str(x)
# print 
dob= x[x.find('birthDate'):].split('"')[2]
print dob


# new_x=len(x)
# for t in x:
	# print t.find_previous_sibling().a['href']
# print x
