from util import *

# Add your import statements here
from collections import defaultdict
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle


class InformationRetrieval():

	def __init__(self, type, lsa_components):
		self.index = None        # inverted index for the corpus
		self.word_order = None	 # (list) word order in the vector notation
		self.doc_vecs = None     # (dict) numpy arrays as vec. representations for the documents
		self.idfs = None         # dictionary of idf value for each word in the corpus
		self.type = type         # (string) could be 'naive'/'vsm'/'lsa'/'gvsm'/'hybrid'
		self.lsa_components = lsa_components

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		if self.type in ['naive', 'vsm']:
			index = defaultdict(set)
			for i in range(len(docIDs)):
				for sentence in docs[i]:
					for token in sentence:
						index[token].add(docIDs[i])
							
			self.index = index

		if self.type == 'vsm':
			self.idfs = get_idfs(self.index, len(docIDs))
			self.word_order = list(self.idfs.keys())
			self.doc_vecs = get_doc_vecs(docs, docIDs, self.word_order, self.idfs)

		elif self.type == 'lsa':
			try:
				with open("saved_params/tf_idf_vectorizer.pkl", 'rb') as f:
					self.vectorizer = pickle.load(f)
				f.close()
				with open("saved_params/tf_idf_docs.pkl", 'rb') as f:
					tf_idf_docs = pickle.load(f)
				f.close()
				
			except:
				self.vectorizer = TfidfVectorizer(stop_words='english')
				tf_idf_docs = self.vectorizer.fit_transform(docs)
				with open("saved_params/tf_idf_vectorizer.pkl", 'wb') as f:
					pickle.dump(self.vectorizer, f)
				f.close()
				with open("saved_params/tf_idf_docs.pkl", 'wb') as f:
					pickle.dump(tf_idf_docs, f)
				f.close()

			self.svd = TruncatedSVD(n_components=self.lsa_components, random_state=42)
			vecs = self.svd.fit_transform(tf_idf_docs)

			self.doc_vecs = {}
			for i in range(len(docIDs)):
				self.doc_vecs[docIDs[i]] = vecs[i] 

		elif self.type == 'gvsm' or self.type == 'hybrid':
			try:
				with open("saved_params/tf_idf_vectorizer.pkl", 'rb') as f:
					self.vectorizer = pickle.load(f)
				f.close()
				with open("saved_params/tf_idf_docs.pkl", 'rb') as f:
					tf_idf_docs = pickle.load(f)
				f.close()
				
			except:
				self.vectorizer = TfidfVectorizer(stop_words='english')
				tf_idf_docs = self.vectorizer.fit_transform(docs)
				with open("saved_params/tf_idf_vectorizer.pkl", 'wb') as f:
					pickle.dump(self.vectorizer, f)
				f.close()
				with open("saved_params/tf_idf_docs.pkl", 'wb') as f:
					pickle.dump(tf_idf_docs, f)
				f.close()
			
			try:
				with open("saved_params/titj.pkl", 'rb') as f:
					self.titj = pickle.load(f)
				f.close()
			except:
				vocab = list(self.vectorizer.vocabulary_.keys())
				vocab.sort(key=lambda x: self.vectorizer.vocabulary_[x])
				syns = get_synsets(vocab)
				self.titj = create_titj(syns)
				with open("saved_params/titj.pkl", 'wb') as f:
					pickle.dump(self.titj, f)
				f.close()

			try:
				with open("saved_params/gvsm_doc_vecs.pkl", 'rb') as f:
					self.doc_vecs = pickle.load(f)
				f.close()
			except:
				self.doc_vecs = {}
				for i in range(len(docIDs)):
					self.doc_vecs[docIDs[i]] = np.dot(r_doc_vec(tf_idf_docs[i]), self.titj)
				with open("saved_params/gvsm_doc_vecs.pkl", 'wb') as f:
					pickle.dump(self.doc_vecs, f)
				f.close()

		if self.type == 'hybrid':
			try:
				with open("saved_params/hybrid_svd.pkl", 'rb') as f:
					self.svd = pickle.load(f)
				f.close()
				with open("saved_params/hybrid_doc_vecs.pkl", 'rb') as f:
					self.doc_vecs = pickle.load(f)
				f.close()
			except:
				doc_matrix = np.array([self.doc_vecs[docIDs[i]] for i in range(len(docIDs))])
				self.svd = TruncatedSVD(n_components=7184, random_state=42)
				vecs = self.svd.fit_transform(doc_matrix)
				with open("saved_params/hybrid_svd.pkl", 'wb') as f:
					pickle.dump(self.svd, f)
				f.close()

				self.doc_vecs = {}
				for i in range(len(docIDs)):
					self.doc_vecs[docIDs[i]] = vecs[i]
				with open(f"saved_params/hybrid_doc_vecs.pkl", 'wb') as f:
					pickle.dump(self.doc_vecs, f)
				f.close()


	
	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		
		if self.type == 'naive':   # naive method based on word counts
			for query in queries:
				doc_counts = defaultdict(lambda: 0)
				for sentence in query:
					for token in sentence:
						for docID in self.index[token]:
							doc_counts[docID] += 1

				rel_docs = list(doc_counts.keys())
				rel_docs.sort(key = lambda x: doc_counts[x], reverse=True)

				doc_IDs_ordered.append(rel_docs)

		elif self.type == 'vsm':
			for query in queries:
				# creating the query unit vector in the vec. space
				q_vec = get_unit_vec(query, self.word_order, self.idfs)
				# getting cosine similarity values
				doc_rel = []
				for d in self.doc_vecs.keys():
					d_vec = self.doc_vecs[d]
					sim = np.dot(d_vec, q_vec)
					doc_rel.append((d, sim))

				doc_rel.sort(key=lambda x: x[1], reverse=True)
				ordered = [d[0] for d in doc_rel]
				doc_IDs_ordered.append(ordered)

		elif self.type == 'lsa':
			tf_idf_queries = self.vectorizer.transform(queries)
			transformed_queries = self.svd.transform(tf_idf_queries)
			for q_vec in transformed_queries:
				doc_rel = []
				for d in self.doc_vecs.keys():
					d_vec = self.doc_vecs[d]
					sim = np.dot(d_vec, q_vec)/(np.linalg.norm(d_vec))/(np.linalg.norm(q_vec))
					doc_rel.append((d, sim))

				doc_rel.sort(key=lambda x: x[1], reverse=True)
				ordered = [d[0] for d in doc_rel]
				doc_IDs_ordered.append(ordered)

		elif self.type == 'gvsm':
			tf_idf_queries = self.vectorizer.transform(queries)

			for query in tf_idf_queries:
				q_vec = np.dot(r_doc_vec(query), self.titj)
				doc_rel = []
				for d in self.doc_vecs.keys():
					d_vec = self.doc_vecs[d]
					sim = gvsm_similarity(d_vec, q_vec)
					doc_rel.append((d, sim))

				doc_rel.sort(key=lambda x: x[1], reverse=True)
				ordered = [d[0] for d in doc_rel]
				doc_IDs_ordered.append(ordered)

		elif self.type == 'hybrid':
			pass

		return doc_IDs_ordered




