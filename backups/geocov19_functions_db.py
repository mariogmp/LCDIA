#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import geocov19_functions_util as geoutil


# In[4]:


def create_db(collection, json_files):
     
    for file in json_files: 
        print('Criando tweets do arquivo '+file)
        with open(file) as json_file: 
            tweets = json.load(json_file)  
            for tweet in tweets:
                date = geoutil.str_to_datetime(tweet['created_at'])
                tweet['created_at'] = date
                tweet['period'] = str(date.year) + "_" + str(date.month).zfill(2)
                tweet['text'] = None
                tweet['lang'] = None
                tweet['score'] = None
                collection.insert_one(tweet)


# In[5]:


def update_tweets_db(collection, json_files):
    
    for file in json_files:
    
        print('Atualizando tweets do arquivo '+file)
        for line in open(file, 'r'):
            tweet = json.loads(line)
            collection.update_one({"tweet_id": tweet['id']}, {'$set':{"text": tweet['full_text'], "lang": tweet['lang']}})


# In[6]:


def create_index(collection, column):
    collection.create_index(column)

