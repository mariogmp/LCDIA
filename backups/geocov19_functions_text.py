#!/usr/bin/env python
# coding: utf-8

# In[13]:


import nltk
import re

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


# In[14]:


# Função auxiliar para remover caracteres indesejados nos textos dos Tweets
chars=["\"","!","@","#","$","%","&","*","(",")","-","_","`","'","{","[","]","}","^","~",",",".",";",":","\","," "]

def clean_words(words):  
    new_words = []    
    for word in words:
         # Verificando se o caracter pertence ao ASCII
        if(all(ord(char) < 128 for char in word)):
            for letter in word:
                if letter in chars:
                    word=word.replace(letter,"")
        new_words.append(word)            
    return new_words


# In[15]:


# Função auxiliar para remover urls nos textos dos Tweets
def clean_urls(words):   
    new_words = []    
    for word in words:              
        if (bool(re.match('http', word)) == False) and (bool(re.match('//tco', word)) == False) and (bool(re.match('//t.co', word)) == False) and (bool(re.match('RT', word)) == False):
            new_words.append(word)
    return new_words


# In[16]:


# Função para remover "stopwords" dos textos dos Tweets
def remove_stopwords(words, language): 
    stopwords = nltk.corpus.stopwords.words(language) + list(punctuation)    
    new_words=[]
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


# In[17]:


# Realiza o processo de limpeza do texto
def process_tweet(text, language):
    words = text.split()
    words = clean_words(words)
    words = clean_urls(words)
    words = remove_stopwords(words, language)
    return words

