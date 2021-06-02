from util import *

# Add your import statements here
from nltk import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for sentence in text:
			curr_sentence = sentence.split(' ')
			tokenized_sentence = []
			for word in curr_sentence:
				if len(word) > 0:
					if word[-1] == ',':
						tokenized_sentence.extend([word[:-1], word[-1]])
					elif word[-2:] == "'s":
						tokenized_sentence.extend([word[:-2], word[-2:]])
					else:
						tokenized_sentence.append(word)

			tokenizedText.append(tokenized_sentence)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		tknizer = TreebankWordTokenizer()
		for sentence in text:
			tokenizedText.append(tknizer.tokenize(sentence))

		return tokenizedText