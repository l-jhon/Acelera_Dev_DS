#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[28]:


import functools
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import seaborn as sns
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, Binarizer, KBinsDiscretizer, MinMaxScaler, StandardScaler, PolynomialFeatures)

import warnings


# In[29]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()

warnings.filterwarnings('ignore')


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[6]:


countries['Country'].values


# In[7]:


countries['Region'].values


# In[8]:


sorted(countries['Region'].unique())


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    return sorted(countries['Region'].unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


def q2():
    countries['Pop_density'] = countries['Pop_density'].str.replace(',','.').astype(float)
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    discretizer.fit(countries[['Pop_density']])
    score_bins = discretizer.transform(countries[['Pop_density']])
    countries['score_bins'] = score_bins
    percentile = np.percentile(score_bins, 90)
    count_countries = len(countries[countries['score_bins'] >percentile])
    return count_countries
    


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[11]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[37]:


def q3():
    count_columns_actual = len(countries.columns)
    countries['Climate'] = countries['Climate'].astype(str)
    countries_one_hot_encoding = pd.get_dummies(countries, columns=['Region', 'Climate'])
    countries_one_hot_encoding = pd.concat([countries, countries_one_hot_encoding])
    return len(countries_one_hot_encoding.columns) - len(countries.columns)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[38]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[44]:


countries.dtypes


# In[63]:


def q4():
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('standard_scaler', StandardScaler())
    ])
    pipeline_transformation = num_pipeline.fit_transform(countries[['Population', 'Area', 'GDP']])

    ## Test_Country já está padronizado, por isso é necessário criar outro pipeline
    num_pipeline_test_country = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    pipeline_transformation_test_country = num_pipeline_test_country.fit_transform(pd.DataFrame(test_country[2:]))
    return round(pipeline_transformation_test_country[9][0],3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[84]:


outilers_Net_migration = countries.Net_migration.copy() # Copiando valores da coluna Net_migration
outilers_Net_migration = outilers_Net_migration.str.replace(',', '.').astype(float) # Retirando vírgula e transformando em float
outilers_Net_migration.fillna(0, inplace=True)


# In[86]:


sns.distplot(outilers_Net_migration);


# In[88]:


outilers_Net_migration.mean()


# In[89]:


outilers_Net_migration.std()


# In[ ]:





# In[90]:


def q5():
    outilers_Net_migration = countries.Net_migration.copy() # Copiando valores da coluna Net_migration
    outilers_Net_migration = outilers_Net_migration.str.replace(',', '.').astype(float) # Retirando vírgula e transformando em float
    outilers_Net_migration.fillna(0, inplace=True)
    
    q1 = outilers_Net_migration.quantile(0.25) # Calculando o primeiro intervalo
    q3 = outilers_Net_migration.quantile(0.75) # Calculando o último intervalo
    iqr = q3 - q1

    outliers_abaixo = outilers_Net_migration < q1 - 1.5 * iqr # observações consideradas outliers abaixo do intervalo permitido
    outliers_acima = outilers_Net_migration > q3 + 1.5 * iqr # observações consideradas outliers acima do intervalo permitido
    
    return (outliers_abaixo.sum(), outliers_acima.sum(), False)


# ## Questão 6
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[337]:


def q6():
    categories = ["sci.electronics", "comp.graphics", "rec.motorcycles"]

    newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    
    word_idx = count_vectorizer.vocabulary_.get('phone')

    return newsgroups_counts[:, word_idx].toarray().sum()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[14]:


def q7():
    categories = ["sci.electronics", "comp.graphics", "rec.motorcycles"]

    newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)

    word_idx = count_vectorizer.vocabulary_.get('phone')

    tfidf_transformer = TfidfTransformer()

    tfidf_transformer.fit(newsgroups_counts)

    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)
    
    return round(newsgroups_tfidf[:,word_idx].toarray().sum(),3)

