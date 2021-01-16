#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import numpy as np
import nltk
import spacy
import collections

from datetime import datetime
from datetime import timedelta

from nltk.tokenize import TweetTokenizer
from string import punctuation
from wordcloud import WordCloud, STOPWORDS

from statistics import mean


# In[136]:


nltk.download('stopwords')
nlp = spacy.load("pt_core_news_sm")


# In[137]:


default_hashtags = ['#coronavírus','#covid19','#covid2019','#covid19brasil','#covid2019brasil','#covid','#corona','#coronavirusbrasil', '#coronavirusnobrasil', '#coronavirus', '#covid-19', '#covidー19', '#covid_19', '#novocoronavírus']


# Funções gerais

# In[138]:


## função para geração de nuvens de palavras a partir de uma lista de strings
def generate_word_cloud(words_list, lower_case):
    
    if (len(words_list) > 0):   
        words=""

        # Criando string a partir das palavras        
        for word in words_list:
            if (lower_case):
                words = words + ' ' + word.lower()
            else:
                words = words + ' ' + word
                 
        stopwords = set(STOPWORDS)
        stopwords.update(nltk.corpus.stopwords.words('portuguese'))
        stopwords.update(['coronavíru','coronaviru','víru','viru','corona','coronavírus','coronavirus','virus','vírus'])
        stopwords.update( ['…','``','...','\'\'','t','https','http','co','rt','pra','pro','vc','pq','q','contra','tudo',',sobre','aí','outro','tá'])
        
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(words)

        plt.figure(figsize=(8,4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


# In[139]:


# Função para montar um df de Quantificadores por data a partir do df de Tweets
def building_dates_df(df, column):

    date_set = set()  
    
    for item in df.created_at:
        date = datetime(item.year, item.month, item.day)
        date_set.add(date)
    
    # Listas utilizadas para montagem do df
    date_list = list(date_set)
    date_list.sort()
    score_mean_list = []
        
    # Populando quantificadores para cada dia
    for index in date_list:
        # Média de scores do período
        score_mean = mean(df[column].loc[(df['created_at'].dt.year == index.year) & (df['created_at'].dt.month == index.month) & (df['created_at'].dt.day == index.day)])
        score_mean_list.append(score_mean)
    
    # Dicionário utilizado como parâmetro para a montagem
    data={'created_at': pd.Series(date_list), column + '_mean':pd.Series(score_mean_list)}
    
    # Criando df
    df = pd.DataFrame(data)
    
    return df 


# In[140]:


# Função para geração de gráficos
def generate_graphic(x, y, label, color, xLabel, yLabel, title, start_period, first_case, first_death):
    
    fig, ax = plt.subplots()

    ax.plot(x,y,color=color, linestyle='solid', linewidth=2,label=label)
    ax.grid()
    ax.margins(0) # remove default margins (matplotlib verision 2+)
    
    first_case_date = datetime.strptime(first_case, '%Y-%m-%d')
    first_death_date = datetime.strptime(first_death, '%Y-%m-%d')
    
    ax.axvspan(datetime.strptime(start_period, '%Y-%m-%d'), 
               datetime.strptime("2020-05-01", '%Y-%m-%d'), 
               fill=True, linewidth=0, color='gainsboro')
    
    plt.axvline(first_case_date, color='orange')
    plt.axvline(first_death_date, color='red')
    
    plt.rcParams['figure.figsize'] = (18,5)
    
    plt.xticks(x, rotation=80)
    plt.legend(fontsize=15)
    plt.xlabel(xLabel,fontsize=15)
    plt.ylabel(yLabel,fontsize=15)
    plt.title(title)
    plt.grid(True, linestyle='--')

    plt.show()


# In[157]:


from datetime import datetime

# Função para geração de gráficos
def generate_graphic_cases(x, y, label, color, xLabel, yLabel, title, restriction, first_case, first_death):
    
    fig, ax = plt.subplots()

    ax.plot(x,y,color=color, linestyle='solid', linewidth=2,label=label)
    ax.grid()
    ax.margins(0) # remove default margins (matplotlib verision 2+)
    
    first_case_date = datetime.strptime(first_case, '%Y-%m-%d')
    first_death_date = datetime.strptime(first_death, '%Y-%m-%d')
    
    if (restriction != None):
        restriction_date = datetime.strptime(restriction, '%Y-%m-%d')
        plt.axvline(restriction_date, color='red')
    
    ax.axvspan(first_case_date, 
               first_death_date,
               fill=True, linewidth=0, color='moccasin')
    
    ax.axvspan(first_death_date, 
               datetime.strptime("2020-05-01", '%Y-%m-%d'), 
               fill=True, linewidth=0, color='gainsboro')
    
    plt.rcParams['figure.figsize'] = (18,5)
    
    plt.xticks(x, rotation=80)
    plt.legend(fontsize=15)
    plt.xlabel(xLabel,fontsize=15)
    plt.ylabel(yLabel,fontsize=15)
    plt.title(title)
    plt.grid(True, linestyle='--')

    plt.show()


# In[142]:


# Função para gerar um gráfico X outro
def generate_vs_graphic(x, y1, y2, label1, label2, color1, color2, xLabel, yLabel, title):
      
    plt.rcParams['figure.figsize'] = (18,5)
    plt.plot(x,y1,color=color1, linestyle='solid', linewidth=2,label=label1)
    plt.plot(x,y2,color=color2, linestyle='solid', linewidth=2,label=label2)

    plt.xticks(x, rotation=80)
    plt.legend(fontsize=15)
    plt.xlabel(xLabel,fontsize=15)
    plt.ylabel(yLabel,fontsize=15)
    plt.title(title)
    plt.grid(True, linestyle='--')
    plt.show()


# In[143]:


def generate_tweets_tokens(texts):
        
    tokens = []
    
    hashtags_words = ['coronavíru','coronaviru','víru','viru','corona','coronavírus','coronavirus','virus','vírus','covid','covid19','covid-19', '19']
    words = ['’','“','','…','``','...','\'\'','t','https','http','co','rt','pra','pro','vc','pq','q','p','contra','tudo','sobre','aí','outro','tá','vai','ser','estar','está','to']
    stopwords = words + default_hashtags + hashtags_words + list(punctuation) + nltk.corpus.stopwords.words('portuguese')
    
    tweet_tokenizer = TweetTokenizer()
    
    for text in texts:        
        words = tweet_tokenizer.tokenize(text)
        pos = nltk.pos_tag(words)
        for word in words:
            word.encode("ascii", errors="ignore").decode()      
            if word.lower() not in stopwords:
                tokens.append(word.lower())
   
    return tokens


# Funções para geração de gráficos

# In[158]:


def generate_bar_from_tokens_freq(tokens_freq, max_res, color, x, y, title):
    
    if len(tokens_freq) > 0:           
        df_words = pd.DataFrame(tokens_freq, columns=['column','total'])
        df_words = df_words.sort_values(by = ['total'], ascending=[False])
        df_words[:max_res].plot(kind='barh', x='column',y='total', figsize=(x, y), color=color, title=title)


# In[159]:


def return_max_phrases_from_interval(df, start, end, max_res):
    
    top_words = []    
    df_query = df.loc[(df['score'] > start) & (df['score'] < end)]

    phrases = list(df_query['text'])   
    counter = collections.Counter(phrases)
    
    return list(counter.most_common(max_res))


# In[145]:


def return_max_tokens_from_interval(df, start, end, max_tokens):
    
    top_words = []    
    df_query = df.loc[(df['score'] > start) & (df['score'] < end)]

    words = list(df_query['text'])
    tokens = generate_tweets_tokens(words)    
    counter = collections.Counter(tokens)
    
    return list(counter.most_common(max_tokens))


# In[146]:


def return_tokens_from_interval(df, start, end):
    
    df_query = df.loc[(df['score'] > start) & (df['score'] < end)]

    words = list(df_query['text'])
    tokens = generate_tweets_tokens(words)
    
    return tokens


# In[147]:


def generate_cloud_from_tokens(tokens, color):

    df_words = pd.DataFrame(tokens, columns=['word'])

    df_freq = df_words['word'].value_counts(normalize = True)
    wordcloud = WordCloud(background_color=color, max_words=100, normalize_plurals=False).generate_from_frequencies(df_freq.to_dict())

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[148]:


def generate_bar_from_tokens(tokens, max_res, color, x, y, title):

    if len(tokens) > 0:
        df_words = pd.DataFrame(tokens, columns=['column'])
        df_words['column'].value_counts()[:max_res].plot(kind='barh', figsize=(x, y), color=color, title=title)


# In[149]:


def generate_bar_from_filtered_tokens(tokens, max_res, word_filter, color, x, y, title):
    
    hashtags = [word for word in tokens if word_filter in word and word not in default_hashtags]
    
    if len(hashtags) > 0:
        df_hashtags = pd.DataFrame(hashtags, columns = ['column'])
        df_hashtags['column'].value_counts()[:max_res].plot(kind='barh', figsize=(x, y), color=color, title=title)


# In[150]:


# Realizando o merge do dataframe de scores x dataframe de casos
def merge_dfs(df_scores_city, df_cases_city):

    # Padronizando as colunas de datas
    df_scores_city = df_scores_city.rename(columns={'created_at':'date'})

    # Convertendo as colunas de datas para o mesmo tipo de objeto
    df_scores_city = df_scores_city.astype({'date': str})
    df_cases_city = df_cases_city.astype({'date': str})

    # Realizando merges para recuperar período de datas em comum
    return pd.merge(df_scores_city, df_cases_city, on='date', how='inner')


# In[151]:


def return_tokens(df, column):
    
    all_tokens = []

    for index, row in df.iterrows():
        tokens = row[column]
        for token in tokens:
            all_tokens.append(token)

    return all_tokens


# Funções para cálculos de correlações com casos de Covid

# In[152]:


# Função para derivar novos atributos
def create_attributes(df_merged):
    
    # Somando colunas new_confirmed + new_deaths (notícias ruins do dia)
    df_merged['bad_news'] = df_merged['new_confirmed'] + df_merged['new_deaths']

    # Somando colunas last_available_confirmed + last_available_deaths (notícias ruins totais)
    df_merged['last_bad_news'] = df_merged['last_available_confirmed'] + df_merged['last_available_deaths']

    # Calculando percentual de crescimento de score de sentimento em relação ao dia anterior
    # df_merged = calculate_percent(df_merged, 'score_mean', 'score_mean_pct')

    # Calculando percentual de crescimento diário de novos casos em relação ao dia anterior
    df_merged = calculate_percent(df_merged, 'new_confirmed', 'new_confirmed_pct')

    # Calculando percentual de crescimento diário de novas mortes em relação ao dia anterior
    df_merged = calculate_percent(df_merged, 'new_deaths', 'new_deaths_pct')

    # Calculando percentual de crescimento diário de novos casos em relação ao total
    df_merged = calculate_percent(df_merged, 'last_available_confirmed', 'last_available_confirmed_pct')

    # Calculando percentual de crescimento diário de novas mortes em relação ao total
    df_merged = calculate_percent(df_merged, 'last_available_deaths', 'last_available_deaths_pct')

    # Calculando percentual de crescimento diário de novas mortes por 100k habitantes em relação ao total
    df_merged = calculate_percent(df_merged, 'last_available_confirmed_per_100k_inhabitants', 'last_available_confirmed_per_100k_inhabitants_pct')

    # Calculando percentual de crescimento diário da taxa de mortalidade em relação ao total
    df_merged = calculate_percent(df_merged, 'last_available_death_rate', 'last_available_death_rate_pct')

    # Calculando percentual de crescimento de notícias ruins em relação ao dia anterior
    df_merged = calculate_percent(df_merged, 'bad_news', 'bad_news_pct')

    # Calculando percentual de crescimento de notícias ruins em relação ao total
    df_merged = calculate_percent(df_merged, 'last_bad_news', 'last_bad_news_pct')
    
    return df_merged;


# In[153]:


## Cálculo do percentual de crescimento de um valor no tempo
def calculate_percent(df, column, new_column):  
    
    values_list = []
    
    is_first = True
    last_value = 0
    
    for (i, row) in df.iterrows():
        if (is_first == True):
            is_first = False
            last_value = row[column]
            values_list.append(0)
        else:
            try:
                percent = 100 * (((row[column] - last_value))/last_value)
                percent = round(percent, 2)
                values_list.append(percent)
                last_value = row[column]
            except ZeroDivisionError:
                values_list.append(0)
                last_value = row[column]
            
    df[new_column] = pd.Series(values_list, index = df.index)
    
    return df


# In[154]:


def prepare_data(df_cases_city, df_scores_city):
    
    # Realizando o merge dos dataframes de média de scores x dataframe de casos
    df_merged = merge_dfs(df_scores_city, df_cases_city)
    #print(df_merged.shape[0])

    # Criando atributos derivados
    df_merged = create_attributes(df_merged)
    
    return df_merged

