from util import get_relevances, get_dcg

# Add your import statements here




class Evaluation():

	def __init__(self):
		"""
		A dictionary of dictionaries which stores doc relevances for each document 
		w.r.t. each query. 
		The value is filled whenever qrels is provided
		"""
		self.relevances = None

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		rel_ret = 0  # no. of relevant docs retrieved

		for docID in query_doc_IDs_ordered[:k]:
			if docID in true_doc_IDs:
				rel_ret += 1

		precision = rel_ret/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		if self.relevances is None:
			self.relevances = get_relevances(qrels)

		sum_prec = 0
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = list(self.relevances[query_id].keys())
			sum_prec += self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)

		meanPrecision = sum_prec/len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		n_rel = len(true_doc_IDs)
		ret_rel = 0    # no. of relevant documents retrieved
		for docID in query_doc_IDs_ordered[:k]:
			if docID in true_doc_IDs:
				ret_rel += 1

		recall = ret_rel/n_rel

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		if self.relevances is None:
			self.relevances = get_relevances(qrels)

		sum_Recall = 0
		for i, query_id in enumerate(query_ids):	
			true_doc_IDs = list(self.relevances[query_id].keys())
			sum_Recall += self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)

		meanRecall = sum_Recall/len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		if precision == 0 or recall == 0:
			fscore = 0
		else:
			fscore = 2*precision*recall/(precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		if self.relevances is None:
			self.relevances = get_relevances(qrels)

		sum_Fscore = 0
		for i, query_id in enumerate(query_ids):			
			true_doc_IDs = list(self.relevances[query_id].keys())
			sum_Fscore += self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)

		meanFscore = sum_Fscore/len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		query_rel = self.relevances[query_id]

		ideal_rel = list(query_rel.values())
		ideal_rel.sort(reverse=True)
		ideal_rel = ideal_rel[:k]

		rel = [0]*k
		for i in range(k):
			doc = query_doc_IDs_ordered[i]
			if doc in true_doc_IDs:
				rel[i] = query_rel[doc] 
			
		dcg = get_dcg(rel)
		# idcg = get_dcg(sorted(rel, reverse=True)) 
		idcg = get_dcg(ideal_rel)
		nDCG = dcg/idcg

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		if self.relevances is None:
			self.relevances = get_relevances(qrels)

		sum_NDCG = 0	
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = list(self.relevances[query_id].keys())
			sum_NDCG += self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)

		meanNDCG = sum_NDCG/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		prec_sum = 0
		count = 0
		for i in range(1, k+1):
			if query_doc_IDs_ordered[i-1] in true_doc_IDs:
				prec_sum += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
				count += 1

		if count == 0:
			avgPrecision = 0
		else:
			avgPrecision = prec_sum/count

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		if self.relevances is None:
			self.relevances = get_relevances(qrels)

		sum_AvgPrec = 0
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = list(self.relevances[query_id].keys())
			sum_AvgPrec += self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)

		meanAveragePrecision = sum_AvgPrec/len(query_ids)

		return meanAveragePrecision

