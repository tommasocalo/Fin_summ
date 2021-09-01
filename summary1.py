#!/usr/bin/env python
# coding: utf-8

# In[11]:



from numpy import dot
from numpy.linalg import norm
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[12]:
if __name__ == "__main__":

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

    def getqns(df):
        low = np.quantile(df, 0.33)
        med = np.quantile(df, 0.66)
        return [low,med]

    #cosine similarity
    def cos_sim(a,b):
        return np.abs(np.round(dot(a, b)/(norm(a)*norm(b)),2))


    # In[13]:


    import pickle
    with open('rank_new1.p', 'rb') as fp:
        rank = pickle.load(fp)
        fp.close()
    with open('Results/Total/res.txt', 'rb') as fp:
        data = eval(fp.read())
    stocks = set(data.keys()).intersection(set(rank.keys()))


    # In[14]:


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


    # In[15]:


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


    # In[16]:


    #virtuous stock has mean over third quartile
    virt = {}
    for col in mean_virt_df.columns:
        qt = getqq(mean_virt_df[col][mean_virt_df[col].notnull()])
        virt[col] = mean_virt_df[col][mean_virt_df[col]>qt[2]].index.values.tolist()


    # In[17]:


    indexes = ['EBITDA','ROE','ROA','R_D', 'Net Income']

    #to calculate quantiles of similarity i take all stocks (NxN matrix)
    dfs_tot = {}
    for ind in indexes:
        dfs_tot[ind] =pd.DataFrame(columns=data.keys(),index = data.keys())
        for stock in data.keys():
            arr = np.array(list(data[stock]))
            for col in dfs_tot[ind].columns:
                if col!=stock:
                    dfs_tot[ind].loc[stock][col] = cos_sim(arr,np.array(list(data[col]))).round(2)

    #to generate summary i take all stocks as row and virtuous stock as column(NxN matrix)
    dfs = {}
    for ind in indexes:
        dfs[ind] =pd.DataFrame(columns=virt[ind],index = data.keys())
        for stock in data.keys():
            arr = np.array(list(data[stock]))
            for col in dfs_tot[ind].columns:
                if col!=stock and col in virt[ind] :
                    dfs[ind].loc[stock][col] = cos_sim(arr,np.array(list(data[col]))).round(2)


    # In[21]:


    getq(dfs_tot['EBITDA'])


    # In[8]:


    q_ind = [0.2, 0.38]


    # In[22]:


    to_exc = pd.DataFrame(columns=['Summary', 'Stock','Index', 'Summarizer','T1','T2','T3','T4','T5','T6'])
    proto = ['Stock ',' is, ','to most of the most virtuous stocks, for the ',' indicator.']
    summ = ['not similar ', 'discreetly similar ','very similar ']
    rs = {}
    for stock in data.keys():
        rs[stock] = {}
        for inn in dfs.keys():
            rs[stock][inn] = np.zeros(3)
            rs[stock][inn][0] = dfs[inn].loc[stock][dfs[inn].loc[stock]<q_ind[0]].size / dfs[inn].loc[stock].size
            rs[stock][inn][1] = dfs[inn].loc[stock][(dfs[inn].loc[stock]>q_ind[0]) & (dfs[inn].loc[stock]<q_ind[1])].size / dfs[inn].loc[stock].size
            rs[stock][inn][2] = dfs[inn].loc[stock][dfs[inn].loc[stock]>q_ind[1]].size / dfs[inn].loc[stock].size

    for stock in data.keys():
        for ind in indexes:

            ress = np.argmax(rs[stock][ind])
            summary = proto[0]+stock+proto[1]+summ[ress]+proto[2]+ind+proto[3]
            t2 = (1-(np.prod(rs[stock][ind])**1/float(rs[stock][ind].shape[0]))).round(2)
            t3 = rs[stock][ind][ress].round(2)
            rk=np.zeros(3)
            dn = 0
            for k,v in rs.items():
                if (not np.array_equal(rs[k],np.zeros(3))) & (ind in rs[k].keys()):
                    dn +=1
                    rk[0] = rk[0] + rs[k][ind][0]
                    rk[1] = rk[1] + rs[k][ind][1]
                    rk[2] = rk[2] + rs[k][ind][2]
            rk = rk/dn
            pr = rk.prod()
            t4 = (t3-pr).round(2)
            r1 = 0.02
            r2 = 0.15
            rss = t3
            res = 0
            if r1<rss<(r1+r2)/2:
                res = 2*(((rss-r1)/(r2-r1))**2)
            if (r1+r2)/2<rss<r2:
                res = 1-2*(((r2-rss)/(r2-r1))**2)
            if rss>r2:
                res = 1
            t5 = res
            t6 = 2*((0.5)**1)

            to_exc = to_exc.append({'Summary': summary, "Stock": stock, "Index": ind,"Summarizer":summ[ress],
                          "T2":t2,"T3":t3,'T5':t5,'T6':t6}, ignore_index=True)


    # In[32]:


    to_exc.shape


    # In[10]:


    to_exc.to_excel("summary1.xlsx")


    # In[ ]:




