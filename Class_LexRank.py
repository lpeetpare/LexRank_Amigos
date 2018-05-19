# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:58:52 2018

@author: garnetliam.peetpare
"""

import numpy as np
import os
import spacy
import re
nlp = spacy.load('en_core_web_lg')
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer


os.chdir('F:/Python Learning')

# Reading in our data
sample_doc = open("doc.txt")
doc = sample_doc.read()

class LexRank:
    def __init__(
            self,
            vectorizer = 'spacy',
            ):
     
        if vectorizer == "spacy": 
            import spacy
            nlp = spacy.load('en_core_web_lg')
            self.vectorizer = self.spacy_vectorizer
        else: 
            raise ValueError("\'spacy\' embeddings are currently the only type of vectorizer available")
# =============================================================================
#         elif vectorizer == "tfidf":
#            from sklearn.feature_extraction.text import TfidfVectorizer
#            self.vectorizer = tfidf_vectorizer
#         else: 
#             raise ValueError("'vectorizer' should be 'spacy' or 'tfidf'") 
# =============================================================================
        
    
    def get_summary(
            self,
            document,
            threshold = 0.8,
            continuous = False,
            damping_factor = 0.2,
            summary_length = 2,
        ):
        '''This function returns a numpy array object containing a summary of the document(s)'''
        if not isinstance(summary_length, int) or summary_length < 1: 
            raise ValueError('\'summary_length\' should be a positive integer') 
        
        if not (
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1)',
            )   
            
        ranked_sentences = self.rank_sentences(document, threshold = threshold, continuous = continuous, damping_factor = damping_factor)  
        summary = ranked_sentences[0:summary_length]
        return summary
         
      
         
            
    def rank_sentences(
            self,
            document,
            threshold = 0.8,
            continuous = False,
            damping_factor = 0.2,
        ): 
        if not (
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1)',
            )   
            
        '''Returns a numpy array of all of the sentences in the document(s) ordered according to their rank of importance'''
        sentences = self.tokenize(document) 
        sentences = np.array(sentences)
        eigenvector = self.power_method(document, threshold = threshold, continuous = continuous, d = damping_factor)
        scores = eigenvector/eigenvector.max()
        ranks = np.argsort(scores)
        ranks = ranks[::-1]
        ranked_sentences = sentences[ranks]
        
        return ranked_sentences
    
        
    
    def power_method(
        self,
        document,
        threshold = 0.8,
        continuous = False,
        d = 0.2,
    ):
        '''Calculates the eigenvector p in the equation p = B^T*p and returns this eigenvector.
        This is the stationary distribution of the Markov chain.'''
        adj_mat = self.adjacency_matrix(document, threshold = threshold, continuous = continuous)
        B = adj_mat/adj_mat.sum(axis=1, keepdims=True)
        eigenvector = np.ones(len(adj_mat)) * 1/(len(adj_mat))
        U = np.ones(shape=(len(adj_mat), len(adj_mat))) * 1/(len(adj_mat))
        transpose = (d*U + (1-d)*B).transpose()
        
        while True:
            eigenvector_next = np.dot(transpose, eigenvector)
    
            if np.allclose(eigenvector_next, eigenvector):
                return eigenvector_next
            else:
                eigenvector = eigenvector_next
        
    
    def adjacency_matrix(
        self,
        document,
        threshold = 0.8,
        continuous = False,
    ): 
        '''Calculates the adjacency matrix of the cosine similarity graph. The threshold specifies the value of cosine
        similarity under which edges are pruned, while setting the continuous argument to "True" means that no pruning occurs.
        Thus, continuous = True is equivalent to threshold = 0. Note the continuous trumps the threshold argument.'''
        cosine_matrix = self.cosine_similarity(document)
        adjacency_matrix = np.zeros(shape=(len(cosine_matrix), len(cosine_matrix)))
        if continuous is True:
            adjacency_matrix = cosine_matrix
        else:
            for i in range(0, len(cosine_matrix)):
                for j in range(0, len(cosine_matrix)):
                    if cosine_matrix[i, j] > threshold:
                        adjacency_matrix[i, j] = 1
                    else:
                        adjacency_matrix[i, j] = 0
        return adjacency_matrix

    
    def cosine_similarity(
        self,
        document,
    ):
        '''Calculates the cosine similarity matrix for the sentences in the document(s).'''
        doc_matrix = self.vectorizer(document)
        cosine_matrix = 1 - pairwise_distances(doc_matrix, metric = "cosine")
        return cosine_matrix
        
    


    def spacy_vectorizer(
        self,
        document,
    ):
      '''Computes the SpaCy embeddings of the sentnces in the document and returns a numpy array
      of these embeddings.'''
      sentences = self.tokenize(document)
      vectors = [nlp(sentence).vector for sentence in sentences]
      return vectors
  
# =============================================================================
#     def tfidf_vectorizer(
#         self,
#         document,
#     ):
#       '''Computes the SpaCy embeddings of the sentnces in the document and returns a numpy array
#       of these embeddings.'''
#       clean_doc = self.clean_corpus(document)
#       vectors = TfidfVectorizer(clean_doc)
#       return vectors
# =============================================================================

    def tokenize(
        self,
        document
    ):
        '''Cleans and tokenizes the document(s) by sentence'''
        clean_doc = self.clean_corpus(document)
        sentences = clean_doc.split(".")
        while '' in sentences:
            sentences.remove('')
        return sentences
    

    def clean_corpus(
        self,
        document,
    ):
        '''Clean the document(s) by removing punctuation and changing to lowercase'''
        corpus_clean = re.sub('[^a-zA-Z0-9\s\.]+', '', document).lower()
        corpus_clean = re.sub("\s{2,}|\n"," ",corpus_clean)
        return corpus_clean



  
Lex = LexRank(vectorizer = "spacy")  

lex_corpus_clean = Lex.clean_corpus(macron)      
lex_sentences = Lex.tokenize(macron)            
lex_vecs = Lex.vectorizer(macron)
lex_cossim_mat = Lex.cosine_similarity(macron)
lex_test = Lex.adjacency_matrix(macron, 0.8, continuous = True)
lex_check = Lex.power_method(macron, 0.8, continuous = True, d = 0.2)

lex_summs = Lex.get_summary(macron, 0.8, continuous = True, damping_factor = 0.2, summary_length = 6)
print(lex_summs)


for i in range(0,34):
    print(np.array_equal(vecs[i], lex_vecs[i]))

     
atai = Lex.get_summary(doc, threshold = 0.7, continuous = False, damping_factor = 0.05, summary_length = 1)  
print(atai)

