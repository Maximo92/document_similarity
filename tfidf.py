#!/usr/bin/env python
#author:	Massimo Innamorati
#date:	07/04/2016
#This program calculates the similarity between a query and a corpus of documents.
#Each line in the input is interpreted as a document. The first document is assumed to be the query.
import sys
from math import log, sqrt

#Ideally use nltk.word_tokenize(), but that was not allowed on Hackerrank.
#Performs a simple tokenisation of text data.
def word_tokenize(data):
	data = data.replace("."," .").replace("\n","")
	return data.split(' ')

#Creates a dictionary of words and their IDs, which are needed for the vector representations.
def make_word_ids(data):
	word_id = {}
	c = 0
	for word in set([word for doc in data for word in doc]):
		word_id[word] = c
		c += 1
	return word_id

#Calculates Term Frequency (i.e. frequency of a term in a document).
def tf(word, doc):
	return float(len([w for w in doc if w == word]))/len(doc)

#Calculates Inverse Document Frequency (i.e. number of documents divided by number of documents with term of interest).
def idf(word, data):
	return log(float(len(data))/len([doc for doc in data if word in doc]))

#Creates a word vector representation for a document in a corpus.        
def make_vector(doc, word_id, data):
	vector = [0.0 for n in range(len(word_id))]
	for word in doc:
		vector[word_id[word]] = tf(word,doc)*idf(word,data)
	return vector

#Ideally use numpy arrays for cleaner performance, but this was not allowed on Hackerrank.
#Calculates the cosine similarity between two word vectors.
def cosine_sim(vec1, vec2):
	similarity = 0
	norm1 = sqrt(sum([n*n for n in vec1]))
	norm2 = sqrt(sum([n*n for n in vec2]))
	for i in range(len(vec1)):
		similarity += vec1[i] * vec2[i]
	return similarity/(sqrt(norm1)*sqrt(norm2))

#Returns the index of the document with the highest cosine similarity to the query vector.
def make_query(query, vectors):
	most_similar = 0
	result = 0
	for i in range(len(vectors)):
		candidate = cosine_sim(query,vectors[i])
		if candidate > most_similar:
			most_similar = candidate
			result = i
	return result+2

if __name__ == '__main__':
	data = [word_tokenize(doc.lower()) for doc in sys.stdin.readlines()]
	word_id = make_word_ids(data)
	vectors = [make_vector(doc, word_id, data) for doc in data]
	result = make_query(vectors[0], vectors[1:])
	sys.stdout.write("%i" % (result))