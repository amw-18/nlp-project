from util import *

# Add your import statements here
from nltk.stem import WordNetLemmatizer



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		#Fill in code here
		lemmatizer = WordNetLemmatizer()

		for sentence in text:
			lemmatized_tokens = []
			for token in sentence:
				lemmatized_tokens.append(lemmatizer.lemmatize(token))

			reducedText.append(lemmatized_tokens)

		
		return reducedText


