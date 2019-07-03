#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    return len(black_friday[black_friday['Age'] == '26-35'][black_friday['Gender'] == 'F'])
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[5]:


def q3():
    return len(pd.unique(black_friday['User_ID']))
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[6]:


def q4():
    return pd.unique(black_friday.dtypes).shape[0]
    pass


# ## Questão 5
# 
# Qual porcentagem das colunas possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[7]:


def q5():
    return max(black_friday.isnull().sum() / len(black_friday))
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[8]:


def q6():
    return max(black_friday.isnull().sum())
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[31]:


def q7():
    return black_friday['Product_Category_3'].value_counts().index[0]
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[12]:


def q8():
    min_normalize = black_friday['Purchase'].min()
    max_normalize = black_friday['Purchase'].max()
    black_friday['Purchase_Normalize'] = (black_friday['Purchase'] - min_normalize) / (max_normalize - min_normalize)
    return float(black_friday['Purchase_Normalize'].mean())
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[15]:


def q9():
    mean_standardize = black_friday['Purchase'].mean()
    std_standardize = black_friday['Purchase'].std()
    black_friday['Purchase_Standardize'] = (black_friday['Purchase'] - mean_standardize) / std_standardize 
    return int(black_friday['Purchase_Standardize'][(black_friday['Purchase_Standardize'] >= -1) & (black_friday['Purchase_Standardize'] <= 1)].count())
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[44]:


def q10():
    validacao = True
    for pd_2, pd_3 in zip(black_friday['Product_Category_2'].values, black_friday['Product_Category_3'].values):
        if pd_2 == None and pd_3 != None:
            validacao = False
    return validacao
    pass


# In[ ]:




