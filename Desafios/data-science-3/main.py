#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
pd.set_option('display.max_columns', 500)

from loguru import logger


# In[24]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[25]:


fifa = pd.read_csv("fifa.csv")


# In[26]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[27]:


# Sua análise começa aqui.
fifa.head()


# In[28]:


# Verificando a quantidade de colunas que possuem valores nulos e suas quantidades
fifa.isnull().sum()


# In[29]:


# Dropando linhas que possuem valores nulos

fifa.dropna(axis=0, inplace=True)


# In[58]:


# Calculando a fração da variância
pca = PCA().fit(fifa)

fracao_variancia = pca.explained_variance_ratio_

fracao_variancia


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[31]:


def q1():
    pca = PCA().fit(fifa)
    fracao_variancia = pca.explained_variance_ratio_
    return round(fracao_variancia[0], 3)


# In[32]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[39]:


def q2():
    pca = PCA(n_components=0.95) # especificando a fração de variância
    qtd_componentes_necessarios = pca.fit_transform(fifa).shape[1]
    return qtd_componentes_necessarios


# In[40]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? Não esqueça de centralizá-lo. Responda como uma tupla de float arredondados para três casas decimais.

# In[67]:


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


# In[23]:



vals_std = fifa - fifa.mean()


# In[70]:


x = np.array(x).reshape(1, -1)


# In[91]:


def q3():
    pca = PCA(n_components=2)

    projected = pca.fit(fifa-fifa.mean())

    pca_x = projected.transform(x)
    
    pca_x = np.around(pca_x[0], 3)
    
    return tuple(pca_x)


# In[92]:


q3()


# In[172]:


fifa.head()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[184]:


def q4():
    Y = fifa.Overall.values
    df = fifa.drop('Overall', axis=1)
    X = df.as_matrix()
    col_names = df.columns

    model_lr = LinearRegression(normalize=True)
    rfe = RFE(model_lr,n_features_to_select=5)
    fit = rfe.fit(X,Y)
    fit.n_features_

    var_list = []

    for r, n in zip(fit.ranking_, col_names):
        if r == 1:
            var_list.append(n)

    return var_list 


# In[ ]:




