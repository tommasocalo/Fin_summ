#!/usr/bin/env python
# coding: utf-8

# In[1]:



from numpy import dot
from numpy.linalg import norm
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:

if __name__ == "__main__":
    const_df = pd.read_csv("data/constituents_csv.csv")
    const_df.head()


    # In[3]:


    sect = {k: list(v) for k, v in const_df.groupby('Sector')['Symbol']}


    # In[4]:


    #get quartiles helper function
    def getqq(df):
        low = np.quantile(df, 0.25)
        med = np.quantile(df, 0.5)
        high = np.quantile(df, 0.75)

        return [low,med,high]

    #get terziles helper function
    def getq(df):
        low = np.quantile(df.stack(), 0.33)
        med = np.quantile(df.stack(), 0.66)
        return [low,med]

    #cosine similarity
    def cos_sim(a,b):
        return np.abs(np.round(dot(a, b)/(norm(a)*norm(b)),2))


    # In[5]:


    import pickle
    with open('rank_new1.p', 'rb') as fp:
        rank = pickle.load(fp)
        fp.close()
    with open('Results/Total/res.txt', 'rb') as fp:
        data = eval(fp.read())

    stocks = set(data.keys()).intersection(set(rank.keys()))


    # In[6]:


    #transform virtuosity into continuous ranking
    cont_virt = {}
    first = True
    for s in stocks:
        cont_virt[s]={}
        for t in rank[s].keys():
            cont_virt[s][t] = {}
            for k in rank[s][t]['year']['very_virtuous']:
                cont_virt[s][t][k] = 1
            for k in rank[s][t]['year']['virtuous']:
                cont_virt[s][t][k] = 0.75
            for k in rank[s][t]['year']['less_virtuous']:
                cont_virt[s][t][k] = 0.5
            for k in rank[s][t]['year']['not_virtuous']:
                cont_virt[s][t][k] = 0.25


    # In[7]:


    #mean virtuosity for each stock, for each indicator
    mean_virt = {}
    for k in cont_virt.keys():
        mean_virt[k] = {}
        for ke in cont_virt[k].keys():
            if len(cont_virt[k][ke].values())>0:
                mean_virt[k][ke] = (sum(cont_virt[k][ke].values())/len(cont_virt[k][ke].values()))

    mean_virt_df = pd.DataFrame(index =mean_virt.keys() )
    for k,v in mean_virt.items():
        for ke, va in v.items():
            mean_virt_df.loc[k,ke] = va


    # In[8]:


    #virtuous stock has mean over third quartile
    virt = {}
    for col in mean_virt_df.columns:
        qt = getqq(mean_virt_df[col][mean_virt_df[col].notnull()])
        virt[col] = mean_virt_df[col][mean_virt_df[col]>qt[2]].index.values.tolist()


    # In[9]:


    indexes = ['EBITDA','ROE','ROA','R_D', 'Net Income']


    # In[10]:


    #similarity quartiles
    q_ind = [0.2, 0.38]


    # In[11]:


    to_exc = pd.DataFrame(columns=['Summary', 'Sector1', 'Sector2','Index', 'Summarizer','T1','T2','T3','T4','T5','T6'])
    proto = ['Most of the stocks of ',' sector, has been ','to most of to the most virtuous stocks of ', ' sector, for ', ' indicator.' ]
    summa = ['not similar ', 'discreetly similar ','very similar ']

    dfs = {}
    rs = {}
    for sect1 in list(sect.keys()):
        dfs[sect1] = {}
        rs[sect1] = {}

        for sect2 in list(sect.keys()):
            if sect1!=sect2:
                dfs[sect1][sect2] = {}
                for ind in indexes:
                    rows = list(set(virt[ind]).intersection(set(sect[sect1])).intersection(stocks))
                    cols = list(set(sect[sect2]).intersection(data.keys()))
                    dfs[sect1][sect2][ind] =pd.DataFrame(columns=cols,index = rows)
                    for row in rows:
                        arr = np.array(list(data[row]))
                        for col in cols:
                            dfs[sect1][sect2][ind].loc[row][col] = cos_sim(arr,np.array(list(data[col]))).round(2)
                rs[sect1][sect2] = {}
                for indd in dfs[sect1][sect2].keys():
                    if dfs[sect1][sect2][indd].mean(axis=1).size>0:
                        rs[sect1][sect2][indd] = np.zeros(3)
                        rs[sect1][sect2][indd][0] = dfs[sect1][sect2][indd].stack()[dfs[sect1][sect2][indd].stack()<q_ind[0]].count() / dfs[sect1][sect2][indd].size
                        rs[sect1][sect2][indd][1] = dfs[sect1][sect2][indd].stack()[(dfs[sect1][sect2][indd].stack()>q_ind[0]) & (dfs[sect1][sect2][indd].stack()<q_ind[1])].count() / dfs[sect1][sect2][indd].size
                        rs[sect1][sect2][indd][2] = dfs[sect1][sect2][indd].stack()[dfs[sect1][sect2][indd].stack()>q_ind[1]].count() / dfs[sect1][sect2][indd].size

    for sect1 in list(sect.keys()):
        for sect2 in list(sect.keys()):
            if sect1!=sect2:
                for ind in indexes:
                    if ind in rs[sect1][sect2].keys():

                        ress = np.argmax(rs[sect1][sect2][ind])
                        summ = proto[0]+sect1+proto[1]+summa[ress]+proto[2]+sect2+proto[3]+ind+proto[4]
                        t2 = (1-(np.prod(rs[sect1][sect2][ind])**1/float(rs[sect1][sect2][ind].shape[0]))).round(2)
                        t3 = rs[sect1][sect2][ind][ress].round(2)
                        rk=np.zeros(3)
                        dn = 0
                        for k,v in rs.items():
                            if (k!=sect2):
                                if (not np.array_equal(rs[k][sect2],np.zeros(3))) & (ind in rs[k][sect2].keys()):
                                    dn +=1
                                    rk[0] = rk[0] + rs[k][sect2][ind][0]
                                    rk[1] = rk[1] + rs[k][sect2][ind][1]
                                    rk[2] = rk[2] + rs[k][sect2][ind][2]
                        rk = rk/dn
                        pr = rk.prod()
                        t4 = (t3-pr).round(2)
                        r1 = 0.02
                        r2 = 0.15
                        rss = t3
                        if rss<r1:
                            res = 0
                        if r1<rss<(r1+r2)/2:
                            res = 2*(((rss-r1)/(r2-r1))**2)
                        if (r1+r2)/2<rss<r2:
                            res = 1-2*(((r2-rss)/(r2-r1))**2)
                        if rss>r2:
                            res = 1
                        t5 = res
                        t6 = 2*((0.5)**1)
                        to_exc = to_exc.append({'Summary': summ, "Sector1": sect1, "Sector2": sect2, "Index": ind,"Summarizer":summa[ress],
                                  "T2":t2,"T3":t3,'T5':t5,'T6':t6}, ignore_index=True)


    # In[13]:


    to_exc.to_excel("summary2.xlsx")


# In[ ]:




