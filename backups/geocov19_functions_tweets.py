#!/usr/bin/env python
# coding: utf-8

# In[15]:


import ijson
import json
import os
import zipfile
import numpy as np
import pandas as pd


# In[16]:


# Função para extrair os arquivos zip e realizar a leitura dos arquivos JSON descompactados
def read_files(zip_dir, json_dir, geo_dir, ids_dir, country_code):
    
    print("Extraindo arquivos...")
    extract_files(zip_dir, json_dir)
    
    json_files = list_files(json_dir, ".json")
        
    total_arquivos = len(json_files)
    total_arquivos_processados = 0
    total_tweets_validos = 0
    
    print(str(total_arquivos) + " arquivo(s) extraídos(s)")
    print("Processando arquivo(s) extraído(s)...")
    
    # Percorrendo arquivos do diretório
    for file in json_files:
        
        total_arquivos_processados = total_arquivos_processados + 1
        
        # Lendo o arquivo JSON extraído     
        num_linhas = sum(1 for line in open(file))
        print("Lendo arquivo '"+get_filename(file)+" com "+str(num_linhas)+" linhas")
        new_tweets = read_tweets(file, country_code)
        print("Tweets válidos encontrados: "+str(len(new_tweets)))
        total_tweets_validos = total_tweets_validos + len(new_tweets)
           
        # Criando dataframe de geolocalização de tweets
        df_geo = create_df_tweets(new_tweets)   
            
        # Escrevendo json com com as geolocalizações brasileiras
        filename = get_filename(file)
        csv_path = geo_dir + os.path.sep + filename
        print("-> Gerando arquivo json '"+get_filename(csv_path))
        df_geo.to_json(csv_path, orient='records', force_ascii=False)
        
        # Escrevendo json com 
        output_file_path = ids_dir + os.path.sep + filename + '_ids.csv'
        print("-> Gerando arquivo com ids '"+output_file_path)
        df_geo.to_csv(output_file_path, sep=';',encoding='utf-8', index=False, header=False, columns=['tweet_id'])        
            
        df_geo = None
        print("Tweets válidos até o momento: "+str(total_tweets_validos))
                    


# In[17]:


# Função para extrair os arquivos zip
def extract_files(zip_dir, json_dir):
    
    zips = list_files(zip_dir, ".zip") 
    
    total_arquivos_processados = 0
    total_arquivos = len(zips)
    
    for file in zips:
        
        total_arquivos_processados = total_arquivos_processados + 1
        
        if (is_file_extracted(file, json_dir)):
            print("-> Arquivo ''"+get_filename(file)+ "' já extraído anteriormente"+" ("+str(total_arquivos_processados)+"/"+str(total_arquivos)+")")
        else:
            # Extraindo arquivo zip
            zip = zipfile.ZipFile(file)
            print("-> Extraindo arquivo '"+get_filename(zip.filename)+"' ("+str(total_arquivos_processados)+"/"+str(total_arquivos)+")")
            zip.extractall(json_dir)
            zip.close()


# In[18]:


# Função para a criação de um dataframe a partir dos tweets gerados com os atributos desejados
def create_df_tweets(tweets):
    
    #tweet_columns = ['tweet_id','created_at','user_id','geo_source','country_code','state','county','city']
    tweet_columns = ['tweet_id','created_at','geo_source','state','city']
    df = pd.DataFrame(tweets, columns = tweet_columns)
    
    # Modificando os tipos de colunas para otimização de espaço
    df.tweet_id = df.tweet_id.astype('int64')
    df.state = df.state.astype('category')
    df.city = df.city.astype('category')
    
    # Informando valores nulos
    df.text = np.nan
    df.score = np.nan
    
    return df


# In[19]:


# Função para retornar a localização do tweet a ser considerada (dentre as várias localizações que podem ter sido informadas)
def select_location(tweet, country_code):
    
    if tweet['geo_source'] == 'coordinates':
        return tweet['geo']
    if tweet['geo_source'] == 'place':
        return tweet['place']
    if tweet['geo_source'] == 'user_location':
        return tweet['user_location']
    if tweet['geo_source'] == 'tweet_text':
        for location in tweet['tweet_locations']:
            if location['country_code'] == country_code:
                return location
    else: 
        return {}


# In[20]:


# Função para identificar se o tweet que está sendo lido pertence ao país desejado
def is_valid_tweet(tweet, country_code):
    
    # Verificando preliminarmente se os dados pertencem a outros países diferentes do Brasil
    if tweet['geo_source'] == 'geo' and 'country_code' in tweet['geo'] and tweet['geo']['country_code'] != country_code:
        return False
    if tweet['geo_source'] == 'place' and 'country_code' in tweet['place'] and tweet['place']['country_code'] != country_code:
        return False
    if tweet['geo_source'] == 'user_location' and 'country_code' in tweet['user_location'] and tweet['user_location']['country_code'] != country_code:
        return False
    
    # Verificando se as informações de cidades e estados são nulas
    if tweet['geo_source'] == 'geo' and 'country_code' in tweet['geo'] and tweet['geo']['country_code'] == country_code and 'state' in tweet['geo'] and tweet['geo']['state'] != np.nan and 'city' in tweet['geo'] and tweet['geo']['city'] != np.nan:
        return True
    if tweet['geo_source'] == 'place' and 'country_code' in tweet['place'] and tweet['place']['country_code'] == country_code and 'state' in tweet['place'] and tweet['place']['state'] != np.nan and 'city' in tweet['place'] and tweet['place']['city'] != np.nan:
        return True
    if tweet['geo_source'] == 'user_location' and 'country_code' in tweet['user_location'] and tweet['user_location']['country_code'] == country_code and 'state' in tweet['user_location'] and tweet['user_location']['state'] != np.nan  and 'city' in tweet['user_location'] and tweet['user_location']['city'] != np.nan:
        return True
    
    # Caso as informações de localização não estejam presentes nos atributos 'geo', 'place' e 'user_location'
    if tweet['geo_source'] == 'tweet_text':
        for location in tweet['tweet_locations']:
            if location['country_code'] == country_code:
                if 'state' in location and location['state'] != np.nan and 'city' in location and location['city'] != np.nan:
                    return True
                else:
                    return False

    return False  


# In[21]:


# Função para a criação de um novo registro de tweet com atributos desejados dos tweets do arquivo original
def create_new_tweet(tweet, country_code):
       
    new_tweet = {}
    
    location = select_location(tweet, country_code)
    
    new_tweet['tweet_id'] = tweet['tweet_id']
    new_tweet['created_at'] = tweet['created_at']
    #new_tweet['user_id'] = tweet['user_id']
    new_tweet['geo_source'] = tweet['geo_source']
    new_tweet['country_code'] = (location['country_code'] if 'country_code' in location else None)
    new_tweet['state'] = (location['state'] if 'state' in location else None)
    #new_tweet['county'] = (location['county'] if 'county' in location else None)
    new_tweet['city'] = (location['city'] if 'city' in location else None)
    
    return new_tweet


# In[22]:


# Função para realizar a leitura de tweets de um país desejado a partir do arquivo JSON
def read_tweets(file, country_code):
    
    with open(file, 'r') as f:
    
        # Array de tweets tratados
        new_tweets = []

        # Realizando leitura iterativa do arquivo (todas as colunas selecionadas)
        objects = ijson.items(f, "", multiple_values=True)

        # Selecionando os tweets desejados
        tweets = (item for item in objects if is_valid_tweet(item, country_code))

        for tweet in tweets:
            new_tweets.append(create_new_tweet(tweet, country_code))
            
    return new_tweets   


# In[23]:


def is_file_extracted(file, dir):
    
    filename = get_filename(file)
    filename = filename.replace(".zip",".json")
    filename = dir + os.path.sep + filename
    return os.path.isfile(filename)


# In[24]:


def list_files(dir, type):
    
    caminhos = [os.path.join(dir, nome) for nome in os.listdir(dir)]
    arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    valid_files = [arq for arq in arquivos if arq.lower().endswith(type)]
                
    return sorted(valid_files)


# In[25]:


def get_filename(file):
    
    split = file.split(os.path.sep)
    size = len(split)
    filename = split[size-1]
    return filename

