# Add your import statements here

import math
from collections import defaultdict
from nltk.corpus.reader.wordnet import lch_similarity
import numpy as np 
from nltk.corpus import wordnet as wn

# Add any utility functions here
def get_relevances(qrels):
    """
    creates a relevance dictionary for the queries
    args:
        qrels : list of dicts
    returns:
        relevances : dict of dicts with keys of the parent dict as query_id
    """
    relevances = defaultdict(dict)
    for dic in qrels:
        relevances[int(dic["query_num"])][int(dic["id"])] = int(dic["position"])

    return relevances

def get_dcg(relevance_order):
    """
    Computes the discounted cumulative gain.
    args:
        relevance_order : list containing relevance values (ordered)
    returns:
        dcg : discounted cumulative gain
    """
    dcg = 0
    for i, rel in enumerate(relevance_order):
        dcg += (rel+1)/math.log(i+2, 2)  # added 1 to numerator for smoothing

    return dcg


def get_tfs(doc):
    """
    args:
        doc : A list of lists
    returns:
        tfs : A dictionary of tokens found in the doc and their normalized tf values.
    """

    tfs = defaultdict(lambda: 0)
    doc_size = 0
    # counting term frequencies
    for sentence in doc:
        for token in sentence:
            tfs[token] += 1
        doc_size += len(sentence)
    
    # normalizing
    for token in tfs:
        tfs[token] /= doc_size
        
    return tfs
        

def get_idfs(index, N):
    """
    args :
        index : dict with keys as term and values as set of docs in which the term appears
            N : Total number of doucments

    returns:
         idfs : dict with keys as term and values as the idf value of that term
    """
    idfs = {}
    for term in index.keys():
        idfs[term] = math.log(N/len(index[term]),2)

    return idfs


def get_doc_vecs(docs, docIDs, word_order, idfs):
    """
    Create tf-idf based document vectors.
    args:
        docs : list of documents
        docIDS : list of document ids
        word_order : word order for the vector space
        idfs : idf values for each word computed on the corpus
    returns:
        doc_vecs : dict of docIDs with their corresponding vec. representation
    """

    doc_vecs = {}
    for doc, docID in zip(docs, docIDs):
        doc_vecs[docID] = get_unit_vec(doc, word_order, idfs)

    return doc_vecs

def get_unit_vec(doc, word_order, idfs):
    """
    Computes unit vector for a particular document.
    args:
        doc : a single document 
        word_order : word order for the vector space
        idfs : idf values for each word computed on the corpus 
    returns:
        vec: unit-vector for a document
    """
    vec = np.array([0.0]*len(word_order))
    tfs = get_tfs(doc)
    for i, word in enumerate(word_order):
        tf = tfs.get(word, 0)
        idf = idfs[word]
        vec[i] = tf*idf

    # special case occuring for only docID 471 (empty doc)
    if np.linalg.norm(vec) == 0:
        return vec

    vec = vec/np.linalg.norm(vec)
    return vec


def similarity_score(S1, S2):
    if S1 is None or S2 is None:
        return 0
    elif S1._name.split('.')[1] != S2._name.split('.')[1]:
        return 0
    else:
        sim = wn.lch_similarity(S1, S2)
        if sim is None:
            return 0
        else:
            return sim
    
        
def create_titj(syns):
    n = len(syns)
    titj = [0]*((n**2 - n)//2)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            val = similarity_score(syns[i], syns[j])
            titj[count] = val
            count += 1

    return np.array(titj)

def r_doc_vec(doc_vec):
    n = doc_vec.shape[1]
    r_vec = [0]*((n**2 - n)//2)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            val = doc_vec[0, i] + doc_vec[0, j]
            r_vec[count] = val
            count += 1
    
    return np.array(r_vec)


def get_synsets(vocab):
    syns = []
    for term in vocab:
        syn = wn.synsets(term)
        if syn:
            syns.append(syn[0])
        else:
            syns.append(None)

    return syns

def gvsm_similarity(d_vec, q_vec):
    d_norm = np.linalg.norm(d_vec)
    q_norm = np.linalg.norm(q_vec)
    return np.dot(q_vec, d_vec)/d_norm/q_norm



