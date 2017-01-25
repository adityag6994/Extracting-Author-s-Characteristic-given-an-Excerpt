import xmltodict
import re ,os

def read_pan(rootdir):
	# import xml.etree.ElementTree as ET
	boredom=0
	refresher=0

	for subdir, dirs, files in os.walk(rootdir):
		pass

	age_group =[];gender = []; data =[]

	for name in files:
		boredom+=1
		if boredom > 10000:
			print refresher
			refresher+=1
			boredom=0

		#print name

		with open(rootdir+'/'+name) as fd:
		    doc = xmltodict.parse(fd.read())

		# print doc['author']['@age_group']
		for con in doc['author']['conversations']['conversation']:
			try:
				if doc['author']['conversations']['@count']=='1':
					notags=re.sub('<[^<]+?>', '',doc['author']['conversations']['conversation']['#text'])
					nocomma = notags.replace(";", "")
					if doc['author']['@age_group']=='10s':
						age_group.append(0)
					elif doc['author']['@age_group']=='20s':
						age_group.append(1)
					else:
						age_group.append(2)
					# print doc['author']['@age_group']
					if (doc['author']['@gender']=='male'):
						gender.append(1)
					else:
						gender.append(0)
					data.append(nocomma)
					# print nocomma
					continue
				# if name == 'c5ceb53928075cf90cf798f6d15e4bff_en_30s_male.xml':
				# 	print doc['author']['conversations']['conversation']['#text']
				notags=re.sub('<[^<]+?>', '',con['#text'])
				nocomma = notags.replace(";", "")
				# print doc['author']['@age_group']

				if doc['author']['@age_group']=='10s':
					age_group.append(0)
				elif doc['author']['@age_group']=='20s':
					age_group.append(1)
				else:
					age_group.append(2)
				if (doc['author']['@gender']=='male'):
					gender.append(1)
				else:
					gender.append(0)
				data.append(nocomma)

			except Exception as e:
				pass
				# print nocomma
			
			# print nocomma
	return data, gender ,age_group

# read_pan()