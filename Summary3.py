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
    sim_q = [0.1,0.22]


    # In[3]:



    #cosine similarity
    def cos_sim(a,b):
        return np.abs(np.round(dot(a, b)/(norm(a)*norm(b)),2))
    def getq(df):
        low = np.quantile(df.stack(), 0.33)
        med = np.quantile(df.stack(), 0.66)
        return [low,med]


    # In[4]:


    import pickle
    with open('Results/Year/res.txt', 'rb') as fp:
        data = eval(fp.read())
    stocks = set(data.keys())
    with open('Results/Total/res.txt', 'rb') as fp:
        datat = eval(fp.read())


    # In[5]:


    from tqdm import tqdm


    # In[91]:


    to_exc = pd.DataFrame(columns=['Summary', 'Stock_1','Stock_2', 'Summarizer','T1','T2','T3','T4','T5','T6'])
    proto = ['In ','the stock ',' has been ','to the stock ']
    summa = ['not similar ', 'discreetly similar ','very similar ']
    quant = ['none ', 'few ','most ', 'all ']
    tw = 'years '


    for stock1 in tqdm(list(stocks)):
        for stock2 in list(stocks):
                dfs =pd.DataFrame(columns=list(data[stock1].keys()),index = [stock2])
                for year in list(data[stock1].keys()):
                    if stock2+'_'+year.split('_')[1] in data[stock2]:
                        arr = np.array(data[stock2][stock2+'_'+year.split('_')[1]]).squeeze()
                        dfs[year] = cos_sim(arr,np.array(data[stock1][year]).squeeze()).round(2)
                rs = np.zeros(3)
                rs[0] = dfs[dfs>sim_q[1]].count().sum()/dfs.size
                rs[1] = dfs[(dfs<sim_q[1])&(dfs>sim_q[0])].count().sum()/dfs.size
                rs[2] = dfs[(dfs<sim_q[0])].count().sum()/dfs.size
                n = [-1, 0.05, 0.05, 0.10]
                m = [0, 0.20, 0.2, 0.45]
                M = [0.3, 0.30, 0.3, 0.8]
                t = [0.65,0.8, 0.8, 1.5]
                r = [n,m,M,t]
                l =[]
                for i in range (4):
                    l.append(fuzz.trapmf(rs, r[i]))
                ress = np.argwhere(l == np.amax(l))

                if ress.size>2:


                    if(np.unique(ress[:,0]).size==1):
                        i = ress[0,0]
                        j = min(ress[:,1])
                    else:
                        i = max(ress[:,0])
                        ag = np.argmax(ress[:,0])
                        j = ress[:,1][ag]
                else:
                    i,j = ress.squeeze()

                t1 = l[i][j].round(2)
                t2 = (1-(np.prod(rs)**(1/float(rs.shape[0])))).round(2)
                t3 = rs[j].round(2)
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

                summary = proto[0]+quant[i]+tw+proto[1]+stock1+proto[2]+summa[2-j]+proto[3]+stock2

                to_exc = to_exc.append({"Summary": summary, "Stock_1": stock1,"Stock_2": stock2, "Summarizer":summa[2-j],                       "Quantifier":quant[i] ,"T1":t1,"T2":t2,"T3":t3,"T5":t5,"T6":t6}, ignore_index=True)


    # In[97]:


    to_exc.to_csv("Summary3.csv")


    # In[ ]:




