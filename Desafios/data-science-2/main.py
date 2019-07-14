#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[7]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[8]:


athletes = pd.read_csv("athletes.csv")


# In[11]:


def get_sample(df, col_name, n=3000, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[13]:


# Sua análise começa aqui.
df = get_sample(athletes,'height')
sct.shapiro(df)


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[44]:


def q1():
    alfa = 0.05
    df = get_sample(athletes,'height')
    nivel = sct.shapiro(df)
    
    if nivel[1] >= alfa:
        return True
    else:
        return False


# In[45]:


q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[46]:


def q2():
    alfa = 0.05
    df = get_sample(athletes,'height')
    nivel = sct.jarque_bera(df)
    
    if int(nivel[1]) >= alfa:
        return True
    else:
        return False


# In[47]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[51]:


def q3():
    alfa = 0.05
    df = get_sample(athletes,'weight')
    norm = sct.normaltest(df)
    
    if norm[1] >= alfa:
        return True
    else:
        return False


# In[52]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[59]:


def q4():
    alfa = 0.05
    df = get_sample(athletes,'weight')
    df = np.log(df)
    norm = sct.normaltest(df)
    
    if norm[1] >= alfa:
        return True
    else:
        return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[116]:


def q5():
    alfa = 0.05
    DataFrame = athletes[(athletes['nationality'] == 'USA') |(athletes['nationality'] == 'BRA') | (athletes['nationality'] == 'CAN')]
    data_one = DataFrame[DataFrame['nationality'] == 'BRA']['height']
    data_two = DataFrame[DataFrame['nationality'] == 'USA']['height']
    th = sct.ttest_ind(data_one, data_two, equal_var=False, nan_policy='omit')
    
    if th[1] >= alfa:
        return True
    else:
        return False


# In[117]:


q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[118]:


def q6():
    alfa = 0.05
    DataFrame = athletes[(athletes['nationality'] == 'USA') |(athletes['nationality'] == 'BRA') | (athletes['nationality'] == 'CAN')]
    data_one = DataFrame[DataFrame['nationality'] == 'BRA']['height']
    data_two = DataFrame[DataFrame['nationality'] == 'CAN']['height']
    th = sct.ttest_ind(data_one, data_two, equal_var=False, nan_policy='omit')
    
    if th[1] >= alfa:
        return True
    else:
        return False


# In[119]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[109]:


def q7():
    alfa = 0.05
    DataFrame = athletes[(athletes['nationality'] == 'USA') |(athletes['nationality'] == 'BRA') | (athletes['nationality'] == 'CAN')]
    data_one = DataFrame[DataFrame['nationality'] == 'USA']['height']
    data_two = DataFrame[DataFrame['nationality'] == 'CAN']['height']
    th = sct.ttest_ind(data_one, data_two, equal_var=False, nan_policy='omit')
    
    return round(th[1],8)


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
