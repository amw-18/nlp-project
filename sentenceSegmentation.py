from util import *

# Add your import statements here
import nltk.data



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		#Fill in code here
		sentenceBoundaries = set(['.', '?','!'])
		sentence = []
		for i in text:
			if i in sentenceBoundaries:
				sentence.append(i)
				segmentedText.append((''.join(sentence)).strip())
				sentence = []
			else:
				sentence.append(i)


		return segmentedText



	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = sent_detector.tokenize(text.strip())

		
		return segmentedText