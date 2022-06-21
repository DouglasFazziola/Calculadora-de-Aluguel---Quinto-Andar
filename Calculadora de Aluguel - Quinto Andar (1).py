#!/usr/bin/env python
# coding: utf-8

# In[229]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as np


# ## Preparando os dados

# In[165]:


df = pd.read_csv('quinto_andar.csv')


# In[166]:


df.head()


# In[167]:


df['quarto'].unique()


# In[168]:


df.loc[df['quarto'] == 'Vila Olímpia, São Paulo']


# In[169]:


df = df.drop(df[df['vaga_carro']=='Jardim Paulista, São Paulo'].index)


# In[170]:


df['quarto'].unique()


# In[171]:


df.drop(['url','total','seguro_incendio','taxa_serviço'], axis=1,inplace=True)


# In[172]:


df['aluguel'] = df['aluguel'].str.slice(2)
df['condominio'] = df['condominio'].str.slice(2)
df['iptu'] = df['iptu'].str.slice(2)
df['quarto'] = df['quarto'].str.slice(0,1)
df['metragem'] = df['metragem'].str.slice(0,-2)
df['banheiro'] = df['banheiro'].str.slice(0,1)
df['andar'] = df['andar'].str.slice(0,-7)
df['vaga_carro'] = df['vaga_carro'].str.slice(0,1)
df['aluguel'] = df['aluguel'].str.replace('.','')
df['vaga_carro'] = df['vaga_carro'].str.replace('S','0')
df['condominio'] = df['condominio'].str.replace('cluso','0')
df['condominio'] = df['condominio'].str.replace('m info','0')
df['iptu'] = df['iptu'].str.replace('cluso','0')


# In[173]:


df['aceita_pet'] = df['aceita_pet'].str.replace('Aceita pet','1')
df['aceita_pet'] = df['aceita_pet'].str.replace('Não aceita','0')
df['mobilia'] = df['mobilia'].str.replace('Sem mobília','0')
df['mobilia'] = df['mobilia'].str.replace('Mobiliado','1')
df['metro_prox'] = df['metro_prox'].str.replace('Metrô próx.','1')
df['metro_prox'] = df['metro_prox'].str.replace('Não próx.','0')


# In[174]:


df.head()


# In[175]:


df['andar'].unique()


# In[176]:


df['andar'][df['andar'] == ''] = '1'


# In[177]:


df['andar'].unique()


# In[178]:


df['condominio'] = df['condominio'].astype(float)
df['iptu'] = df['iptu'].astype(float)
df['metragem'] = df['metragem'].astype(float)
df['quarto'] = df['quarto'].astype(int)
df['banheiro'] = df['banheiro'].astype(int)
df['vaga_carro'] = df['vaga_carro'].astype(int)
df['aceita_pet'] = df['aceita_pet'].astype(int)
df['mobilia'] = df['mobilia'].astype(int)
df['metro_prox'] = df['metro_prox'].astype(int)
df['aluguel'] = df['aluguel'].astype(int)
df['andar'] = df['andar'].astype(int)


# In[179]:


df.info()


# In[180]:


label_bairro = LabelEncoder()


# In[181]:


df['bairro'] = label_bairro.fit_transform(df['bairro'])


# In[182]:


df['bairro'].unique()


# In[183]:


df.corr()['aluguel'].sort_values()


# In[184]:


df.corr()


# In[185]:


ss = StandardScaler()


# In[186]:


x = df.drop('aluguel',axis=1)


# In[187]:


y = df['aluguel']


# In[188]:


x = ss.fit_transform(x)


# In[189]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# ## Regressão Linear Simples

# In[190]:


reg = LinearRegression()


# In[191]:


reg.fit(X_train, y_train)


# In[192]:


p = reg.predict(X_test)


# In[193]:


[mae(y_test,p),
mse(y_test,p)]


# In[194]:


r2_score(y_test,p)


# ## Lasso

# In[195]:


lasso = Lasso()


# In[196]:


lasso.fit(X_train,y_train)


# In[197]:


l = lasso.predict(X_test)


# In[198]:


r2_score(y_test,l)


# ## Arvore de decisão

# In[214]:


from sklearn.tree import DecisionTreeRegressor


# In[215]:


dt = DecisionTreeRegressor()


# In[216]:


dt.fit(X_train,y_train)


# In[217]:


d = dt.predict(X_test)


# In[218]:


r2_score(y_test,d)


# ## RandomForrest

# In[103]:


from sklearn.ensemble import RandomForestRegressor


# In[156]:


rf = RandomForestRegressor(n_estimators=130)


# In[157]:


rf.fit(X_train,y_train)


# In[158]:


r = rf.predict(X_test)


# In[159]:


r2_score(y_test,r)


# In[249]:


mae(y_test,r)


# ## Polinomial

# In[160]:


from sklearn.preprocessing import PolynomialFeatures


# In[243]:


p = PolynomialFeatures()
X_train_pol = p.fit_transform(X_train)
X_test_pol = p.transform(X_test)


# In[244]:


lrp = LinearRegression()
lrp.fit(X_train_pol, y_train)


# In[245]:


ppol = lrp.predict(X_test_pol)


# In[246]:


r2_score(y_test,ppol)


# In[248]:


mae(y_test,ppol)


# 
