#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import pandas as pd
import os
import numpy as np
import seaborn as sns


# In[3]:

if __name__ == "__main__":
    years = {}
    total_dfs = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Results/df')
                 for name in files
                 if name.endswith((".csv"))]
    for f in total_dfs:
        df = pd.read_csv(f)
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        if 'Unnamed: 1' in df.columns:
            df = df.set_index('Unnamed: 1')
        df.index = pd.to_datetime(df.index).year
        name = f.split('/')[-1].split('.')[0]
        years[name] = df.index.unique().values


    # In[4]:


    total_st = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Data/Fundamental_SP500/')
                 for name in files
                 if name.endswith(("st.csv"))]
    total_bs = [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Data/Fundamental_SP500/')
                 for name in files
                 if name.endswith(("bs.csv"))]
    total_cf= [os.path.join(root, name)
                 for root, dirs, files in os.walk(os.getcwd()+'/Data/Fundamental_SP500/')
                 for name in files
                 if name.endswith(("cf.csv"))]


    # In[5]:


    dir_dict = {}
    stocks = set()
    for root, dirs, files in os.walk(os.getcwd()+'/Data/Fundamental_SP500/'):
        for name in files:
            stock = name.split('/')[-1].split('.')[0].split('_')[0]
            if stock in set(years.keys()):
                if stock not in stocks:
                    stocks.add(stock)
                    dir_dict[stock] = {'year':{},'quarter':{}}
                    path_st = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_st.csv'
                    path_st_q = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_st_q.csv'
                    path_bs = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_bs.csv'
                    path_bs_q = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_bs_q.csv'
                    path_cf = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_cf.csv'
                    path_cf_q = os.getcwd()+'/Data/Fundamental_SP500/'+stock+'_cf_q.csv'
                    if os.path.isfile(path_st):
                        dir_dict[stock]['year']['st'] = path_st
                    if os.path.isfile(path_st_q):
                        dir_dict[stock]['quarter']['st_q'] = path_st_q
                    if os.path.isfile(path_bs):
                        dir_dict[stock]['year']['bs'] = path_bs
                    if os.path.isfile(path_bs_q):
                        dir_dict[stock]['quarter']['bs_q'] = path_bs_q
                    if os.path.isfile(path_cf):
                        dir_dict[stock]['year']['cf'] = path_cf
                    if os.path.isfile(path_cf_q):
                        dir_dict[stock]['quarter']['cf_q'] = path_cf_q


    # In[6]:


    for el in set(dir_dict.keys())-set(years.keys()):
        dir_dict.pop(el, None)


    # In[7]:


    #vengono scartati i valori a intersezione non nulla, i titoli con gli indicatori mancanti


    # In[8]:


    ebitda_y_df=pd.DataFrame({})
    missing_ebitda_y_df = []
    for f in stocks:
        df = pd.read_csv(dir_dict[f]['year']['st'])
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        if 'Unnamed: 1' in df.columns:
            df = df.set_index('Unnamed: 1')
        if 'ttm' in df.index:
            df = df.drop('ttm')
        if 'EBITDA' in df:

            df.index = pd.to_datetime(df.index).year.astype(str)

            df = df[~df.index.duplicated(keep='first')]

            df = df.loc[df.index.intersection(pd.Index(years[f].astype(str)))]
            if df.index.intersection(pd.Index(years[f].astype(str))).values.size!=0:
                new_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: df['EBITDA']},index = df.index)
                ebitda_y_df = pd.concat([ebitda_y_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
            else:
                missing_ebitda_y_df.append(f)
        else:
            missing_ebitda_y_df.append(f)
    ebitda_y_df = ebitda_y_df.apply(lambda x: x.apply(lambda t: t.replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ebitda_y_df = ebitda_y_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ebitda_y_df.sort_index(ascending=False,inplace=True)


    # In[10]:


    ebitda_q_df=pd.DataFrame({})
    missing_ebitda_q_df = []
    for f in stocks:
        df = pd.read_csv(dir_dict[f]['quarter']['st_q'])
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        if 'Unnamed: 1' in df.columns:
            df = df.set_index('Unnamed: 1')
        if 'ttm' in df.index:
            df = df.drop('ttm')
        if 'EBITDA' in df:
            if (pd.to_datetime(df.index).year.intersection(pd.Index(years[f]))).values.size!=0:
                df = df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(df.index).year.values ]]
                df.index = pd.PeriodIndex(pd.to_datetime(df.index), freq='Q').astype(str)
                df = df[~df.index.duplicated(keep='first')]
                new_df = pd.DataFrame({f: df['EBITDA']},index = df.index)
                ebitda_q_df = pd.concat([ebitda_q_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
            else:
                missing_ebitda_q_df.append(f)
        else:
            missing_ebitda_q_df.append(f)
    ebitda_q_df = ebitda_q_df.apply(lambda x: x.apply(lambda t: t.replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ebitda_q_df = ebitda_q_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ebitda_q_df.sort_index(ascending=False,inplace=True)


    # In[13]:


    ROE_y_df=pd.DataFrame({})
    missing_ROE_y_df = []
    for f in stocks:
        if ('st' in dir_dict[f]['year']) & ('bs'in dir_dict[f]['year']):
            st_df = pd.read_csv(dir_dict[f]['year']['st'])
            bs_df = pd.read_csv(dir_dict[f]['year']['bs'])
            if 'Unnamed: 0' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 1')
            if 'Unnamed: 0' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 1')
            if 'ttm' in st_df.index:
                st_df = st_df.drop('ttm')
            st_df = st_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ' react-empty: 3 ' in bs_df.columns:
                bs_df = bs_df.drop(' react-empty: 3 ',axis=1)
            bs_df = bs_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ('Total Assets' in bs_df) & ('Net Income' in st_df) & ('Total Liabilities' in bs_df):
                if ((pd.to_datetime(bs_df.index).year.intersection(pd.Index(years[f]))).values.size!=0) and                 ((pd.to_datetime(st_df.index).year.intersection(pd.Index(years[f]))).values.size!=0):
                    bs_df.index = pd.to_datetime(bs_df.index).year.astype(str)
                    bs_df = bs_df[~bs_df.index.duplicated(keep='first')]
                    bs_df = bs_df.loc[bs_df.index.intersection(pd.Index(years[f].astype(str)))]

                    st_df.index = pd.to_datetime(st_df.index).year.astype(str)
                    st_df = st_df[~st_df.index.duplicated(keep='first')]
                    st_df = st_df.loc[st_df.index.intersection(pd.Index(years[f].astype(str)))]

                    ass = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Assets'].astype(float)},index = bs_df.index)
                    lia = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Liabilities'].astype(float)},index = bs_df.index)
                    new_bs_df = ass.subtract(lia)

                    new_st_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: st_df['Net Income'].astype(float)},index = st_df.index)
                    new_roe= new_st_df.div(new_bs_df)
                    ROE_y_df = pd.concat([ROE_y_df,new_roe],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_ROE_y_df.append(f)
            else:
                missing_ROE_y_df.append(f)
        else:
            missing_ROE_y_df.append(f)
    ROE_y_df = ROE_y_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ROE_y_df = ROE_y_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ROE_y_df.sort_index(ascending=False,inplace=True)


    # In[15]:


    ROE_q_df=pd.DataFrame({})
    missing_ROE_q_df = []
    for f in stocks:
        if ('st_q' in dir_dict[f]['quarter']) & ('bs_q'in dir_dict[f]['quarter']):
            st_df = pd.read_csv(dir_dict[f]['quarter']['st_q'])
            bs_df = pd.read_csv(dir_dict[f]['quarter']['bs_q'])
            if 'Unnamed: 0' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 1')
            if 'Unnamed: 0' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 1')
            if 'ttm' in st_df.index:
                st_df = st_df.drop('ttm')
            st_df = st_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ' react-empty: 3 ' in bs_df.columns:
                bs_df = bs_df.drop(' react-empty: 3 ',axis=1)
            bs_df = bs_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ('Total Assets' in bs_df) & ('Net Income' in st_df) & ('Total Liabilities' in bs_df):
                if ((pd.to_datetime(bs_df.index).year.intersection(pd.Index(years[f]))).values.size!=0) and                 ((pd.to_datetime(st_df.index).year.intersection(pd.Index(years[f]))).values.size!=0):

                    bs_df = bs_df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(bs_df.index).year.values ]]
                    st_df = st_df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(st_df.index).year.values ]]

                    bs_df.index = pd.PeriodIndex(pd.to_datetime(bs_df.index), freq='Q').astype(str)
                    bs_df = bs_df[~bs_df.index.duplicated(keep='first')]
                    st_df.index = pd.PeriodIndex(pd.to_datetime(st_df.index), freq='Q').astype(str)
                    st_df = st_df[~st_df.index.duplicated(keep='first')]
                    ass = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Assets'].astype(float)},index = bs_df.index)
                    lia = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Liabilities'].astype(float)},index = bs_df.index)
                    new_bs_df = ass.subtract(lia)
                    new_st_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: st_df['Net Income'].astype(float)},index = st_df.index)
                    new_roe= new_st_df.div(new_bs_df)
                    ROE_q_df = pd.concat([ROE_q_df,new_roe],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_ROE_q_df.append(f)
            else:
                missing_ROE_q_df.append(f)
        else:
            missing_ROE_q_df.append(f)
    ROE_q_df = ROE_q_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ROE_q_df = ROE_q_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ROE_q_df.sort_index(ascending=False,inplace=True)


    # In[25]:


    ROA_y_df=pd.DataFrame({})
    missing_ROA_y_df = []
    for f in stocks:
        if ('st' in dir_dict[f]['year']) & ('bs'in dir_dict[f]['year']):
            st_df = pd.read_csv(dir_dict[f]['year']['st'])
            bs_df = pd.read_csv(dir_dict[f]['year']['bs'])
            if 'Unnamed: 0' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 1')
            if 'Unnamed: 0' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 1')
            if 'ttm' in st_df.index:
                st_df = st_df.drop('ttm')
            st_df = st_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ' react-empty: 3 ' in bs_df.columns:
                bs_df = bs_df.drop(' react-empty: 3 ',axis=1)
            bs_df = bs_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ('Total Assets' in bs_df) & ('Net Income' in st_df):
                if ((pd.to_datetime(bs_df.index).year.intersection(pd.Index(years[f]))).values.size!=0) and                 ((pd.to_datetime(st_df.index).year.intersection(pd.Index(years[f]))).values.size!=0):
                    bs_df.index = pd.to_datetime(bs_df.index).year.astype(str)
                    bs_df = bs_df[~bs_df.index.duplicated(keep='first')]
                    bs_df = bs_df.loc[bs_df.index.intersection(pd.Index(years[f].astype(str)))]

                    st_df.index = pd.to_datetime(st_df.index).year.astype(str)
                    st_df = st_df[~st_df.index.duplicated(keep='first')]
                    st_df = st_df.loc[st_df.index.intersection(pd.Index(years[f].astype(str)))]

                    new_bs_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Assets'].astype(float)},index = bs_df.index)
                    new_st_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: st_df['Net Income'].astype(float)},index = st_df.index)
                    new_roe= new_st_df.div(new_bs_df)
                    ROA_y_df = pd.concat([ROA_y_df,new_roe],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_ROA_y_df.append(f)
            else:
                missing_ROA_y_df.append(f)
        else:
            missing_ROA_y_df.append(f)
    ROA_y_df = ROA_y_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ROA_y_df = ROA_y_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ROA_y_df.sort_index(ascending=False,inplace=True)


    # In[26]:


    ROA_q_df=pd.DataFrame({})
    missing_ROA_q_df = []
    for f in stocks:
        if ('st_q' in dir_dict[f]['quarter']) & ('bs_q'in dir_dict[f]['quarter']):
            st_df = pd.read_csv(dir_dict[f]['quarter']['st_q'])
            bs_df = pd.read_csv(dir_dict[f]['quarter']['bs_q'])
            if 'Unnamed: 0' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in st_df.columns:
                st_df = st_df.set_index('Unnamed: 1')
            if 'Unnamed: 0' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in bs_df.columns:
                bs_df = bs_df.set_index('Unnamed: 1')
            if 'ttm' in st_df.index:
                st_df = st_df.drop('ttm')
            st_df = st_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ' react-empty: 3 ' in bs_df.columns:
                bs_df = bs_df.drop(' react-empty: 3 ',axis=1)
            bs_df = bs_df.apply(lambda x: x.apply(lambda k: str(k).replace(',',''))).replace('',np.nan).replace('-',np.nan)
            if ('Total Assets' in bs_df) & ('Net Income' in st_df):
                if ((pd.to_datetime(bs_df.index).year.intersection(pd.Index(years[f]))).values.size!=0) and                 ((pd.to_datetime(st_df.index).year.intersection(pd.Index(years[f]))).values.size!=0):
                    st_df = st_df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(st_df.index).year.values ]]

                    bs_df = bs_df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(bs_df.index).year.values ]]

                    bs_df.index = pd.PeriodIndex(pd.to_datetime(bs_df.index), freq='Q').astype(str)
                    bs_df = bs_df[~bs_df.index.duplicated(keep='first')]
                    st_df.index = pd.PeriodIndex(pd.to_datetime(st_df.index), freq='Q').astype(str)
                    st_df = st_df[~st_df.index.duplicated(keep='first')]
                    new_bs_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: bs_df['Total Assets'].astype(float)},index = bs_df.index)
                    new_st_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: st_df['Net Income'].astype(float)},index = st_df.index)
                    new_roe= new_st_df.div(new_bs_df)
                    ROA_q_df = pd.concat([ROA_q_df,new_roe],axis=1).replace(np.nan, '', regex=True).replace(',','.')
            else:
                missing_ROA_q_df.append(f)
        else:
            missing_ROA_q_df.append(f)
    ROA_q_df = ROA_q_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ROA_q_df = ROA_q_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ROA_q_df.sort_index(ascending=False,inplace=True)


    # In[27]:


    rd_y_df=pd.DataFrame({})
    missing_rd_y_df = []
    for f in stocks:
        if ('st' in dir_dict[f]['year']):
            df = pd.read_csv(dir_dict[f]['year']['st'])
            if 'Unnamed: 0' in df.columns:
                df = df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in df.columns:
                df = df.set_index('Unnamed: 1')
            if 'ttm' in df.index:
                df = df.drop('ttm')
            if 'Research Development' in df:
                if (pd.to_datetime(df.index).year.intersection(pd.Index(years[f]))).values.size!=0:

                    df.index = pd.to_datetime(df.index).year.astype(str)

                    df = df[~df.index.duplicated(keep='first')]
                    df = df.loc[df.index.intersection(pd.Index(years[f].astype(str)))]

                    new_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: df['Research Development']},index = df.index)
                    rd_y_df = pd.concat([rd_y_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                     missing_rd_y_df.append(f)
            else:
                missing_rd_y_df.append(f)
        else:
            missing_rd_y_df.append(f)
    rd_y_df = rd_y_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    rd_y_df = rd_y_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    rd_y_df.sort_index(ascending=False,inplace=True)


    # In[28]:


    rd_q_df=pd.DataFrame({})
    missing_rd_q_df = []
    for f in stocks:
        if ('st_q' in dir_dict[f]['quarter']):
            df = pd.read_csv(dir_dict[f]['quarter']['st_q'])
            if 'Unnamed: 0' in df.columns:
                df = df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in df.columns:
                df = df.set_index('Unnamed: 1')
            if 'ttm' in df.index:
                df = df.drop('ttm')
            if 'Research Development' in df:
                if (pd.to_datetime(df.index).year.intersection(pd.Index(years[f]))).values.size!=0:
                    df = df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(df.index).year.values ]]

                    df.index = pd.PeriodIndex(pd.to_datetime(df.index), freq='Q').astype(str)
                    df = df[~df.index.duplicated(keep='first')]

                    new_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: df['Research Development']},index = df.index)
                    rd_q_df = pd.concat([rd_q_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_rd_q_df.append(f)
            else:
                missing_rd_q_df.append(f)
        else:
            missing_rd_q_df.append(f)
    rd_q_df = rd_q_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    rd_q_df = rd_q_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    rd_q_df.sort_index(ascending=False,inplace=True)


    # In[29]:


    ni_y_df=pd.DataFrame({})
    missing_ni_y_df = []
    for f in stocks:
        if ('st' in dir_dict[f]['year']):
            df = pd.read_csv(dir_dict[f]['year']['st'])
            if 'Unnamed: 0' in df.columns:
                df = df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in df.columns:
                df = df.set_index('Unnamed: 1')
            if 'ttm' in df.index:
                df = df.drop('ttm')
            if 'Net Income' in df:
                if (pd.to_datetime(df.index).year.intersection(pd.Index(years[f]))).values.size!=0:
                    df.index = pd.to_datetime(df.index).year.astype(str)
                    df = df[~df.index.duplicated(keep='first')]
                    df = df.loc[df.index.intersection(pd.Index(years[f].astype(str)))]

                    new_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: df['Net Income']},index = df.index)
                    ni_y_df = pd.concat([ni_y_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_ni_y_df.append(f)


            else:
                missing_ni_y_df.append(f)
        else:
            missing_ni_y_df.append(f)
    ni_y_df = ni_y_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ni_y_df = ni_y_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ni_y_df.sort_index(ascending=False,inplace=True)


    # In[30]:


    ni_q_df=pd.DataFrame({})
    missing_ni_q_df = []
    for f in stocks:
        if ('st_q' in dir_dict[f]['quarter']):
            df = pd.read_csv(dir_dict[f]['quarter']['st_q'])
            if 'Unnamed: 0' in df.columns:
                df = df.set_index('Unnamed: 0')
            if 'Unnamed: 1' in df.columns:
                df = df.set_index('Unnamed: 1')
            if 'ttm' in df.index:
                df = df.drop('ttm')
            if 'Net Income' in df:
                if (pd.to_datetime(df.index).year.intersection(pd.Index(years[f]))).values.size!=0:
                    df = df.loc[[el in pd.Index(years[f]).values for el in pd.to_datetime(df.index).year.values ]]

                    df.index = pd.PeriodIndex(pd.to_datetime(df.index), freq='Q').astype(str)
                    df = df[~df.index.duplicated(keep='first')]

                    new_df = pd.DataFrame({f.split('/')[-1].split('_')[0]: df['Net Income']},index = df.index)
                    ni_q_df = pd.concat([ni_q_df,new_df],axis=1).replace(np.nan, '', regex=True).replace(',','.')
                else:
                    missing_ni_q_df.append(f)
            else:
                missing_ni_q_df.append(f)
        else:
            missing_ni_q_df.append(f)
    ni_q_df = ni_q_df.apply(lambda x: x.apply(lambda t: str(t).replace(',',''))).replace('',np.nan).replace('-',np.nan)
    ni_q_df = ni_q_df.apply(lambda x: x.apply(lambda z: float(z) if z!=np.nan else z))
    ni_q_df.sort_index(ascending=False,inplace=True)


    # In[31]:


    def normalize(df):
        result = df.copy()
        for feature_name in df.columns:
            mean = df[feature_name].mean()
            std = df[feature_name].std()
            result[feature_name] = (df[feature_name] - mean) / std
        return result
    ebitda_y_df_norm = normalize(ebitda_y_df)
    ebitda_q_df_norm = normalize(ebitda_q_df)
    ROE_y_df_norm = normalize(ROE_y_df)
    ROE_q_df_norm = normalize(ROE_q_df)
    ROA_y_df_norm = normalize(ROA_y_df)
    ROA_q_df_norm = normalize(ROA_q_df)
    rd_y_df_norm = normalize(rd_y_df)
    rd_q_df_norm = normalize(rd_q_df)
    ni_y_df_norm = normalize(ni_y_df)
    ni_q_df_norm = normalize(ni_q_df)


    # In[32]:


    ebitda_y_df_shift= ebitda_y_df_norm.sub(ebitda_y_df_norm.shift())
    ebitda_q_df_shift = ebitda_q_df_norm.sub(ebitda_q_df_norm.shift())
    ROE_y_df_shift = ROE_y_df_norm.sub(ROE_y_df_norm.shift())
    ROE_q_df_shift = ROE_q_df_norm.sub(ROE_q_df_norm.shift())
    ROA_y_df_shift = ROA_y_df_norm.sub(ROA_y_df_norm.shift())
    ROA_q_df_shift = ROA_q_df_norm.sub(ROA_q_df_norm.shift())
    rd_y_df_shift = rd_y_df_norm.sub(rd_y_df_norm.shift())
    rd_q_df_shift = rd_q_df_norm.sub(rd_q_df_norm.shift())
    ni_y_df_shift = ni_y_df_norm.sub(ni_y_df_norm.shift())
    ni_q_df_shift = ni_q_df_norm.sub(ni_q_df_norm.shift())


    # In[33]:


    def reject_outliers(data, m=3):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    ebitda_y_df_no_out =  reject_outliers(ebitda_y_df_shift)
    ebitda_q_df_no_out =  reject_outliers(ebitda_q_df_shift)
    ROE_y_df_no_out =  reject_outliers(ROE_y_df_shift)
    ROE_q_df_no_out = reject_outliers(ROE_q_df_shift)
    ROA_y_df_no_out = reject_outliers(ROA_y_df_shift)
    ROA_q_df_no_out = reject_outliers(ROA_q_df_shift)
    rd_y_df_no_out = reject_outliers(rd_y_df_shift)
    rd_q_df_no_out = reject_outliers(rd_q_df_shift)
    ni_y_df_no_out = reject_outliers(ni_y_df_shift)
    ni_q_df_no_out = reject_outliers(ni_q_df_shift)


    # In[34]:


    def getq(df):
        low = np.quantile(df.stack(), 0.25)
        med = np.quantile(df.stack(), 0.5)
        high = np.quantile(df.stack(), 0.75)
        return [low,med,high]

    ebitda_y_q = getq(ebitda_y_df_shift)
    ebitda_q_q = getq(ebitda_q_df_shift)
    ROE_y_q = getq(ROE_y_df_shift)
    ROE_q_q = getq(ROE_q_df_shift)
    ROA_y_q = getq(ROA_y_df_shift)
    ROA_q_q = getq(ROA_q_df_shift)
    rd_y_q = getq(rd_y_df_shift)
    rd_q_q = getq(rd_q_df_shift)
    ni_y_q = getq(ni_y_df_shift)
    ni_q_q = getq(ni_q_df_shift)


    # In[35]:


    f_dict = {}


    # In[36]:


    def comp(df,name,period,q):
        for col in df.columns:
            if col not in f_dict:
                f_dict[col] = {}
                if name not in f_dict[col]:
                     f_dict[col][name] = {'year':{'very_virtuous':[],'virtuous':[],'less_virtuous':[],'not_virtuous':[]}, 'quarter':{'very_virtuous':[],'virtuous':[],'less_virtuous':[],'not_virtuous':[]}}
                if period == 'year':
                    f_dict[col][name][period]['very_virtuous'] = df[col][df[col]>q[2]].index.values.tolist()
                    f_dict[col][name][period]['virtuous'] = df[col][(df[col]<q[2])&(df[col]>q[1])].index.values.tolist()
                    f_dict[col][name][period]['less_virtuous'] = df[col][(df[col]<q[1])&(df[col]>q[0])].index.values.tolist()
                    f_dict[col][name][period]['not_virtuous'] = df[col][(df[col]<q[0])].index.values.tolist()
                else:
                    f_dict[col][name][period]['very_virtuous'] = df[col][df[col]>q[2]].index.values.astype(str).tolist()
                    f_dict[col][name][period]['virtuous'] = df[col][(df[col]<q[2])&(df[col]>q[1])].index.values.astype(str).tolist()
                    f_dict[col][name][period]['less_virtuous'] = df[col][(df[col]<q[1])&(df[col]>q[0])].index.values.astype(str).tolist()
                    f_dict[col][name][period]['not_virtuous'] = df[col][(df[col]<q[0])].index.values.astype(str).tolist()


            else:
                if name not in f_dict[col]:
                     f_dict[col][name] = {'year':{'very_virtuous':[],'virtuous':[],'less_virtuous':[],'not_virtuous':[]}, 'quarter':{'very_virtuous':[],'virtuous':[],'less_virtuous':[],'not_virtuous':[]}}
                if period == 'year':
                    f_dict[col][name][period]['very_virtuous'] = df[col][df[col]>q[2]].index.values.tolist()
                    f_dict[col][name][period]['virtuous'] = df[col][(df[col]<q[2])&(df[col]>q[1])].index.values.tolist()
                    f_dict[col][name][period]['less_virtuous'] = df[col][(df[col]<q[1])&(df[col]>q[0])].index.values.tolist()
                    f_dict[col][name][period]['not_virtuous'] = df[col][(df[col]<q[0])].index.values.tolist()
                else:
                    f_dict[col][name][period]['very_virtuous'] = df[col][df[col]>q[2]].index.values.astype(str).tolist()
                    f_dict[col][name][period]['virtuous'] = df[col][(df[col]<q[2])&(df[col]>q[1])].index.values.astype(str).tolist()
                    f_dict[col][name][period]['less_virtuous'] = df[col][(df[col]<q[1])&(df[col]>q[0])].index.values.astype(str).tolist()
                    f_dict[col][name][period]['not_virtuous'] = df[col][(df[col]<q[0])].index.values.astype(str).tolist()

    comp(ebitda_y_df_shift,'EBITDA','year',ebitda_y_q)
    comp(ebitda_q_df_shift,'EBITDA','quarter',ebitda_q_q)
    comp(ROE_y_df_shift,'ROE','year',ROE_y_q)
    comp(ROE_q_df_shift,'ROE','quarter',ROE_y_q)
    comp(ROA_y_df_shift,'ROA','year',ROA_y_q)
    comp(ROA_q_df_shift,'ROA','quarter',ROA_q_q)
    comp(rd_y_df_shift,'R_D','year',rd_y_q)
    comp(rd_q_df_shift,'R_D','quarter',rd_q_q)
    comp(ni_y_df_shift,'Net Income','year',ni_y_q)
    comp(ni_q_df_shift,'Net Income','quarter',ni_q_q)




    import json

    with open('rank_new.json', 'w') as fp:
        json.dump(f_dict, fp)


    # In[39]:


    f_dict['T']


    # In[38]:


    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle

    with open('rank_new1.p', 'wb') as fp:
        pickle.dump(f_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


    # In[ ]:




