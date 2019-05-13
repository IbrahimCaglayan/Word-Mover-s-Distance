#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:39:55 2019

@author: ibrahimcaglayan
"""
import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import timeit




#Calculate the Word travel cost
def Word_Travel_Cost(w2v_1,w2v_2):
    return np.sqrt(np.sum(np.square(w2v_1-w2v_2),axis=0))

#Class nBow represent the normalized BOW and sentence (vector of words) corresponding to nbow vector
class nBow:
  def __init__(self,d,sentence):
     self.d = d
     self.sentence = sentence
     
#Calculate the nBow of a sentence
def nBow_D(sentence):
    L=len(sentence)
    d=np.array([])
    sentence_nBow=[]
    for i in range(L):
        if sentence[i] not in sentence_nBow:
            sentence_nBow.append(sentence[i])
            count=0
            for j in range(L):
                if sentence[i]==sentence[j]:
                    count=count+1
            d=np.concatenate((d,count/L),axis=None)
    return nBow(d,sentence_nBow)
          

#Calculate the Words Centroid distance between 2 sentences
def WCD(sentence1,sentence2):
    
    D1=nBow_D(sentence1)
    D2=nBow_D(sentence2)
    
    Xd1=np.zeros(300)
    Xd2=np.zeros(300)
    
    for i in range(len(D1.d)):
        Xd1+=embeddings_index.get(D1.sentence[i])*D1.d[i]
    for j in range(len(D2.d)):
        Xd2+=embeddings_index.get(D2.sentence[j])*D2.d[j]
        
    return np.sqrt(np.sum(np.square(Xd1-Xd2),axis=0))

#Calculate the Relaxed Words Moving Distance between 2 sentences
def RWMD(sentence1,sentence2):
    D1=nBow_D(sentence1)
    D2=nBow_D(sentence2)  
    L1=len(D1.d) #length of nbow 1
    L2=len(D2.d) #length of nbow 2
    LR1=0 #Relaxed solution 1
    LR2=0 #Relaxed solution 2
    #Calculate the Relaxed solution 1
    for i in range(L1):
        w2v_1=embeddings_index.get(D1.sentence[i])
        distances_word_to_sentences=np.zeros(L2)
        for j in range(L2):
            w2v_2=embeddings_index.get(D2.sentence[j])
            distances_word_to_sentences[j]=Word_Travel_Cost(w2v_1,w2v_2)
        min_cost=np.min(distances_word_to_sentences)
        LR1+=min_cost*D1.d[i]
        
    #Calculate the Relaxed solution 2
    for i in range(L2):
        w2v_2=embeddings_index.get(D2.sentence[i])
        distances_word_to_sentences=np.zeros(L2)
        for j in range(L1):
            w2v_1=embeddings_index.get(D1.sentence[j])
            distances_word_to_sentences[j]=Word_Travel_Cost(w2v_1,w2v_2)
        min_cost=np.min(distances_word_to_sentences)
        LR2+=min_cost*D2.d[i]
            
    return np.max([LR1,LR2])
            
 
def Prefetch_and_prune(refDoc, docList, k):
    
    #refDoc is the referent document
    #docList is the list of documents to compare
    #k for the k-nn value
    
    n = len(docList)
    knn = np.zeros((k,3))
    DIC = np.zeros((n,3))
    
    for i in range(0, n):
        
        DIC[i][2] = i
        DIC[i][0] += WCD(refDoc,docList[i])
    #sort list of documents by WCD to referent document
    indSort=np.argsort(DIC,axis=0)
    DIC =  DIC[indSort[:,0],:]
    #calculate wmd for the knn
    for i in range(0, k):      
        knn[i][2] += DIC[i][2]
        knn[i][1] += DIC[i][0]
        idx=DIC[i][2]
        idx=int(idx)
        knn[i][0] += wv_from_bin.wmdistance(refDoc,(docList[idx]))
    
    #Checking lower bound
    for i in range(k,n):
        
        currDoc = docList[int(DIC[i][2])]
        
        diff = knn[k-1][1] - RWMD(refDoc, currDoc)#distance WCD-RWMD
        
        if diff < 0:
            #skip to next iteration
            continue
        #prune up
        wmd_curr=wv_from_bin.wmdistance(refDoc,currDoc)
        DIC[i][1] = wmd_curr
        knn_to_add=[DIC[i][1],DIC[i][0],DIC[i][2]]
        knn = np.append(knn, [knn_to_add],axis=0)
        
        indSort_knn = np.argsort(knn,axis=0)
        knn =  knn[indSort_knn[:,0],:]
        knn = np.delete(knn, -1, 0)

        
        
    K={}
    for i in range(k):
        idx=knn[i][2]
        idx=int(idx)
        K[i]=" ".join(str(x) for x in docList[idx])
        
    return K


filepath = "GoogleNews-vectors-negative300.bin"

embeddings_index = {}# word2vec dictionnary of words

##load word2vec
i=0
wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True) 
for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
    coefs = np.asarray(vector, dtype='float32')
    i=i+1
    embeddings_index[word] = coefs




#input sentences
sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The President greets the press in Chicago'
sentence_president ='The band gave a concert in Japan'

#remove stop words on sentences
sentence_obama = remove_stopwords(sentence_obama)
sentence_president = remove_stopwords(sentence_president)

#
sentence_obama = sentence_obama.lower().split()
sentence_president = sentence_president.lower().split()


##WCD
start = timeit.default_timer()
print('distance wcd',WCD(sentence_obama,sentence_president))
stop = timeit.default_timer()
print('distance wcd Time: ', stop - start)  

##RWMD
start = timeit.default_timer()
print('distance RWMD',RWMD(sentence_obama,sentence_president))
stop = timeit.default_timer()
print('distance RWMD Time: ', stop - start)  

##WMD
start = timeit.default_timer()
print('distance WMD',wv_from_bin.wmdistance(sentence_obama, sentence_president))
stop = timeit.default_timer()
print('distance WMD Time: ', stop - start)  


## Prefetch_and_prune
sref='Obama speaks to the media in Illinois'
s1='The President greets the press in Chicago'
s2='The color of the big car is blue'
s3='Bush talks to journalists in Washington'
s4='The band gave a concert in Japan'

sref= remove_stopwords(sref)
s1=remove_stopwords(s1)
s2=remove_stopwords(s2)
s3=remove_stopwords(s3)
s4=remove_stopwords(s4)


sref= sref.lower().split()
s1 = s1.lower().split()
s2 = s2.lower().split()
s3 = s3.lower().split()
s4 = s4.lower().split()


S_doc=list([s1,s2,s3,s4])

start = timeit.default_timer()
print('Prefetch_and_prune',Prefetch_and_prune(sref, S_doc, 2))
stop = timeit.default_timer()
print('distance Prefetch_and_prune Time: ', stop - start) 



