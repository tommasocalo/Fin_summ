#!/usr/bin/env python
# coding: utf-8

# In[15]:



from numpy import dot
from numpy.linalg import norm
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[16]:
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


    # In[17]:


    import pickle
    with open('rank_new1.p', 'rb') as fp:
        rank = pickle.load(fp)
        fp.close()
    with open('Results/Total/res.txt', 'rb') as fp:
        data = eval(fp.read())
    stocks = set(data.keys()).intersection(set(rank.keys()))


    # In[19]:


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


    # In[20]:


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


    # In[21]:


    #virtuous stock has mean over third quartile
    virt = {}
    for col in mean_virt_df.columns:
        qt = getqq(mean_virt_df[col][mean_virt_df[col].notnull()])
        virt[col] = mean_virt_df[col][mean_virt_df[col]>qt[2]].index.values.tolist()


    # In[22]:


    indexes = ['EBITDA','ROE','ROA','R_D', 'Net Income']

    #to calculate quantiles of similarity i take all stocks (NxN matrix)
    dfs_tot = {}
    for ind in indexes:
        dfs_tot[ind] =pd.DataFrame(columns=stocks,index = data.keys())
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


    # In[28]:


    q_ind=[0.21, 0.38]
    for ind in dfs.keys():
        dfs[ind] = dfs[ind].T


    # In[31]:


    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    pw = list(powerset(range(0,5)))[1:16]


    # In[35]:


    to_exc = pd.DataFrame(columns=['Summary', 'Stock','Index_1','Index_2', 'Summarizer','T1','T2','T3','T4','T5','T6'])
    proto = ['Stock ',' is ','at the ', ' of the most virtuous stocks, for the ',' indicator.']
    summ = ['not similar ', 'discreetly similar ','very similar ']
    for stock in data.keys():
        rs = {tt:np.zeros(3) for tt in pw}
        for t in pw:
            if len(t) == 1:
                rs[t][0]=((dfs[indexes[t[0]]][stock]<=q_ind[0])).mean()
                rs[t][1]=(((dfs[indexes[t[0]]][stock]>q_ind[0]) & (dfs[indexes[t[0]]][stock]<=q_ind[1] ))).mean()
                rs[t][2]=((dfs[indexes[t[0]]][stock]>q_ind[1])).mean()

            if len(t) == 2:
                rs[t][0]=((dfs[indexes[t[0]]][stock]<q_ind[0]) & (dfs[indexes[t[1]]][stock]<q_ind[0])).mean()
                rs[t][1]=(((dfs[indexes[t[0]]][stock]>q_ind[0]) & (dfs[indexes[t[0]]][stock]<q_ind[1] ))             & ((dfs[indexes[t[1]]][stock]>q_ind[0]) & (dfs[indexes[t[1]]][stock]<q_ind[1]))).mean()
                rs[t][2]=((dfs[indexes[t[0]]][stock]>q_ind[1]) & (dfs[indexes[t[1]]][stock]>q_ind[1])).mean()
        for inde in rs.keys():
            if len(inde)==1:
                ress = np.argmax(rs[inde])
                summary = proto[0]+stock+proto[1]+summ[ress]+proto[2]+str(int(rs[inde].round(2)[ress]*100))+'%'+proto[3]+indexes[inde[0]]+proto[4]

                t2 = (1-(np.prod(rs[inde])**(1/float(rs[inde].shape[0])))).round(2)
                t3 = rs[inde][ress].round(2)
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

                to_exc = to_exc.append({"Summary": summary, "Stock": stock, "Index_1": indexes[inde[0]],"Summarizer":summ[ress],                       "T2":t2,"T3":t3,"T5":t5,"T6":t6}, ignore_index=True)

            if len(inde)==2:
                ress = np.argmax(rs[inde])
                if (rs[inde][ress]>0.15):
                    summary = proto[0]+stock+proto[1]+summ[ress]+proto[2]+str(int(rs[inde].round(2)[ress]*100))+'%'+proto[3]+indexes[inde[0]]+' and the '+indexes[inde[1]]+proto[4]
                    t_1, t_2 = (inde[0],),(inde[1],)
                    t2 = (1-(np.prod(rs[inde])**(1/float(rs[inde].shape[0])))).round(2)
                    t3 = rs[inde][ress].round(2)
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
                    t4 = abs(rs[inde][res]-rs[t_1][res]*rs[t_2][res]).round(2)
                    to_exc = to_exc.append({"Summary": summary, "Stock": stock, "Index_1": indexes[inde[0]], "Index_2": indexes[inde[1]],"Summarizer":summ[ress],                       "T2":t2,"T3":t3,"T4":t4,"T5":t5,"T6":t6}, ignore_index=True)


    # In[41]:


    to_exc.to_excel("summary5.xlsx")


    # In[45]:


'''for t in pw:

    if len(t) == 1:
        rs[t][0]=((dfs[indexes[t[0]]][stock]<=q_ind[0])).mean()
        rs[t][1]=(((dfs[indexes[t[0]]][stock]>q_ind[0]) & (dfs[indexes[t[0]]][stock]<=q_ind[1] ))).mean()
        rs[t][2]=((dfs[indexes[t[0]]][stock]>q_ind[1])).mean()

    if len(t) == 2:
        rs[t][0]=((dfs[indexes[t[0]]][stock]<q_ind[0]) & (dfs[indexes[t[1]]][stock]<q_ind[0])).mean()
        rs[t][1]=(((dfs[indexes[t[0]]][stock]>q_ind[0]) & (dfs[indexes[t[0]]][stock]<q_ind[1] )) \
        & ((dfs[indexes[t[1]]][stock]>q_ind[0]) & (dfs[indexes[t[0]]][stock]<q_ind[1]))).mean()
        rs[t][2]=((dfs[indexes[t[0]]][stock]>q_ind[1]) & (dfs[indexes[t[1]]][stock]>q_ind[1])).mean()
    if len(t) == 3:
        rs[t][0]=((dfs[indexes[t[0]]][sto[0]]<q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[0]) \
        & (dfs[indexes[t[2]]][sto[0]]<q_ind[0])).mean()

        rs[t][1]=(((dfs[indexes[t[0]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[0]]][sto[0]]<q_ind[1]) ) \
        & ((dfs[indexes[t[1]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[2]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[2]]][sto[0]]<q_ind[1]))).mean()

        rs[t][2]=((dfs[indexes[t[0]]][sto[0]]>q_ind[1]) & (dfs[indexes[t[1]]][sto[0]]>q_ind[1])\
                & (dfs[indexes[t[2]]][sto[0]]>q_ind[1])).mean()

    if len(t) == 4:
        rs[t][0]=((dfs[indexes[t[0]]][sto[0]]<q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[0])\
        & (dfs[indexes[t[2]]][sto[0]]<q_ind[0]) & (dfs[indexes[t[3]]][sto[0]]<q_ind[0])).mean()

        rs[t][1]=(((dfs[indexes[t[0]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[0]]][sto[0]]<q_ind[1] ))\
        & ((dfs[indexes[t[1]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[2]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[2]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[3]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[3]]][sto[0]]<q_ind[1]))).mean()

        rs[t][2]=((dfs[indexes[t[0]]][sto[0]]>q_ind[1]) & (dfs[indexes[t[1]]][sto[0]]>q_ind[1])\
                & (dfs[indexes[t[2]]][sto[0]]>q_ind[1]) & (dfs[indexes[t[3]]][sto[0]]>q_ind[1])).mean()
    if len(t) == 5:
        rs[t][0]=((dfs[indexes[t[0]]][sto[0]]<q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[0])\
        & (dfs[indexes[t[2]]][sto[0]]<q_ind[0]) & (dfs[indexes[t[3]]][sto[0]]<q_ind[0])\
        & (dfs[indexes[t[4]]][sto[0]]<q_ind[0])).mean()

        rs[t][1]=(((dfs[indexes[t[0]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[0]]][sto[0]]<q_ind[1] ))\
        & ((dfs[indexes[t[1]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[1]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[2]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[2]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[3]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[3]]][sto[0]]<q_ind[1]))\
        & ((dfs[indexes[t[4]]][sto[0]]>q_ind[0]) & (dfs[indexes[t[4]]][sto[0]]<q_ind[1]))).mean()

        rs[t][2]=((dfs[indexes[t[0]]][sto[0]]>q_ind[1]) & (dfs[indexes[t[1]]][sto[0]]>q_ind[1])\
                & (dfs[indexes[t[2]]][sto[0]]>q_ind[1]) & (dfs[indexes[t[3]]][sto[0]]>q_ind[1])\
                & (dfs[indexes[t[4]]][sto[0]]>q_ind[1])).mean()
'''
