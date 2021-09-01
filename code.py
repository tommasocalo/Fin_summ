import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
class SP500_series(object):

        def __init__(self,company):
            self.company = company
            self.sp_path = "Data/SP500/"+company+".csv"
            self.sent_path = "news-sp500/"+company+".csv"
            self.df = pd.read_csv(self.sp_path)
            self.df['Date'] = pd.to_datetime(self.df.Date)
            self.df.set_index('Date', inplace=True)
            self.df_final = pd.DataFrame(index=self.df.index, data={'MAC_SMA5:20':'','MAC_EMA5:20':'',
                                              'MAC_SMA20:50':'','MAC_EMA20:50':'',
                                              'MAC_SMA50:200':'','MAC_EMA50:200':'',
                                              'RSI':'','AO':'', 'OBV':'','PVO':'','ADL':'','MACD':''})
            self.expressions = {}
        def mac_sma(self,a,b):
            s1 = ''
            s2 = ''
            if a == 5:
                s1 = 'A'
                s2 = 'B'
            if a == 20:
                s1  = 'C'
                s2 = 'D'
            if a == 50:
                s1 = 'E'
                s2 = 'F'
            rolling1 = self.df.Close.rolling(str(a)+'D').mean()
            rolling2 = self.df.Close.rolling(str(b)+'D').mean()
            df_trend = pd.DataFrame({'Date':rolling2.index, 'b':rolling2.values,'a':rolling1.values} )
            for i in range (df_trend.index.size-1):
                if (df_trend.loc[i,'a']<df_trend.loc[i,'b']) & (df_trend.loc[i+1,'a']>df_trend.loc[i+1,'b']):
                    df_trend.loc[i+1,'trend']=s1
                elif (df_trend.loc[i,'a']>df_trend.loc[i,'b']) & (df_trend.loc[i+1,'a']<df_trend.loc[i+1,'b']):
                    df_trend.loc[i+1,'trend']=s2
                else:
                    df_trend.loc[i+1,'trend'] =np.nan
            column = 'MAC_SMA'+str(a)+':'+str(b)
            self.df_final[column]= df_trend.trend.values
            self.expressions['A']= "Simple moving average of 5 periods crossed above simple moving average of 20 periods"
            self.expressions['B']= "Simple moving average of 5 periods crossed below simple moving average of 20 periods"
            self.expressions['C']="Simple moving average of 20 periods crossed above simple moving average of 50 periods"
            self.expressions['D']= "Simple moving average of 20 periods crossed below simple moving average of 50 periods"
            self.expressions['E']="Simple moving average of 50 periods crossed above simple moving average of 200 periods"
            self.expressions['F']= "Simple moving average of 50 periods crossed below simple moving average of 200 periods"
            return self.df_final[self.df_final[column].notnull()][column]

        def mac_ema(self,a,b):
            s1 = ''
            s2 = ''
            if a == 5:
                s1 = 'G'
                s2 = 'H'
            if a == 20:
                s1  = 'I'
                s2 = 'J'
            if a == 50:
                s1 = 'K'
                s2 = 'L'
            rolling1 =self.df.Close.ewm(int(a)).mean()
            rolling2 =self.df.Close.ewm(int(b)).mean()
            df_trend = pd.DataFrame({'Date':rolling2.index, 'b':rolling2.values,'a':rolling1.values} )
            for i in range (df_trend.index.size-1):
                if (df_trend.loc[i,'a']<df_trend.loc[i,'b']) & (df_trend.loc[i+1,'a']>df_trend.loc[i+1,'b']):
                    df_trend.loc[i+1,'trend']=s1
                elif (df_trend.loc[i,'a']>df_trend.loc[i,'b']) & (df_trend.loc[i+1,'a']<df_trend.loc[i+1,'b']):
                    df_trend.loc[i+1,'trend']=s2
                else:
                    df_trend.loc[i+1,'trend'] =np.nan
            column = 'MAC_EMA'+str(a)+':'+str(b)
            self.df_final[column]= df_trend.trend.values
            self.expressions['G']="Exponential moving average of 5 periods crossed above simple moving average of 20 periods"
            self.expressions['H']= "Exponential moving average of 5 periods crossed below simple moving average of 20 periods"
            self.expressions['I']="Exponential moving average of 20 periods crossed above simple moving average of 50 periods"
            self.expressions['J']= "Exponential moving average of 20 periods crossed below simple moving average of 50 periods"
            self.expressions['K']="Exponential moving average of 50 periods crossed above simple moving average of 200 periods"
            self.expressions['L']= "Exponential moving average of 50 periods crossed below simple moving average of 200 periods"


            return self.df_final[self.df_final[column].notnull()][column]

        def rsi(self):
            delta = self.df.Close.diff()
            window = 14
            up_days = delta.copy()
            up_days[delta<=0]=0.0
            down_days = abs(delta.copy())
            down_days[delta>0]=0.0
            RS_up = up_days.rolling(window).mean()
            RS_down = down_days.rolling(window).mean()
            rsi= 100-100/(1+RS_up/RS_down)
            df_rsi = pd.DataFrame(index=self.df.index,data={'RSI':rsi.values,'trend':''})

            for i in range (rsi.index.size-1):
                    if (df_rsi['RSI'].iloc[i]<70) & (df_rsi['RSI'].iloc[i+1]>70 ):
                        df_rsi['trend'].iloc[i+1]='M'
                    elif (df_rsi['RSI'].iloc[i]>30) & (df_rsi['RSI'].iloc[i+1]<30 ):
                        df_rsi['trend'].iloc[i+1]='N'
                    else:  df_rsi['trend'].iloc[i+1] =np.nan

            self.df_final['RSI']= df_rsi.trend.values
            self.expressions['M']="RSI crossed above 70%"
            self.expressions['N']= "RSI crossed below 30%"

            return self.df_final[self.df_final['RSI']=='N']['RSI']

        def ao(self):
            periods = 25
            aroon_up = self.df['High'].rolling(periods+1).apply(lambda x: x.argmax(), raw=True) / periods * 100
            aroon_down = self.df['Low'].rolling(periods+1).apply(lambda x: x.argmin(), raw=True) / periods * 100
            df_a = pd.DataFrame(index=self.df.index,data={'aroon':aroon_up-aroon_down,'trend':''})

            for i in range (df_a.index.size-1):
                if (df_a['aroon'].iloc[i]<70) & (df_a['aroon'].iloc[i+1]>70 ):
                    df_a['trend'].iloc[i+1]='Q'
                elif (df_a['aroon'].iloc[i]>30) & (df_a['aroon'].iloc[i+1]<30 ):
                    df_a['trend'].iloc[i+1]='R'
                else:  df_a['trend'].iloc[i+1] =np.nan
            self.df_final['AO']= df_a.trend.values
            self.expressions['Q']="Aroon oscillator crossed above 70%"
            self.expressions['R']= "Aroon oscillator crossed below 30%"

            return self.df_final[self.df_final['AO'].notnull()]['AO']

        def pvo(self):
            exp1 = self.df.Volume.ewm(span=12, adjust=False).mean()
            exp2 = self.df.Volume.ewm(span=26, adjust=False).mean()

            df_pvo = pd.DataFrame({'Date':exp1.index, 'b':exp2.values,'a':exp1.values})
            for i in range (df_pvo.index.size-1):
                if (df_pvo.loc[i,'a']<df_pvo.loc[i,'b']) & (df_pvo.loc[i+1,'a']>df_pvo.loc[i+1,'b']):
                    df_pvo.loc[i+1,'trend']='U'
                elif (df_pvo.loc[i,'a']>df_pvo.loc[i,'b']) & (df_pvo.loc[i+1,'a']<df_pvo.loc[i+1,'b']):
                    df_pvo.loc[i+1,'trend']='V'
                else: df_pvo.loc[i+1,'trend'] =np.nan
            self.df_final['PVO']= df_pvo.trend.values
            self.expressions['U']="Volumes exponential moving average of 12 periods crossed above volumes exponential moving average of 26 periods"
            self.expressions['V']= "Volumes exponential moving average of 12 periods crossed below volumes exponential moving average of 26 periods"

            return self.df_final[self.df_final['PVO'].notnull()]['PVO']

        def sent(self):
            sid = SentimentIntensityAnalyzer()
            df_sent = pd.read_csv(self.sent_path,sep='\t')
            df_sent.ts = df_sent.ts.map(lambda x: re.search('^[0-9]{8}',str(x)).group(0))
            df_sent.ts = df_sent.ts.map(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%d/%m/%Y'))
            df_sent['body_comp'] = df_sent['body'].apply(lambda x:sid.polarity_scores(str(x))['compound'])
            df_sent['title_comp'] = df_sent.title.apply(lambda x:sid.polarity_scores(str(x))['compound'])
            del df_sent['href']
            del df_sent['keyword']

            df_sent['ts'] = pd.to_datetime(df_sent.ts)
            if not df_sent.empty :
                
                df_sent = df_sent.groupby('ts',as_index=False).mean()
            df_sent.set_index('ts', inplace=True)
            df_nan = pd.DataFrame(np.nan, index=self.df.index, columns=['Sentiments'])
            df_sent= df_sent.join(df_nan, how='outer')
            for i in range (df_sent.index.size):
                avg_sent = (float(df_sent.body_comp.iloc[i])+float(df_sent.title_comp.iloc[i]))/2
                if avg_sent > 0:
                    df_sent['Sentiments'].iloc[i]='2'
                elif avg_sent <= 0:
                    df_sent['Sentiments'].iloc[i]='3'
                else:  df_sent['Sentiments'].iloc[i] =np.nan
            del df_sent['body_comp']
            del df_sent['title_comp']
            self.df_final = self.df_final.join(df_sent, how='outer')
            self.expressions['2']="Positive news"
            self.expressions['3']= "Negative news"

            return self.df_final[self.df_final['Sentiments'].notnull()]['Sentiments']

        def ad(self):
            exp1 = self.df.Volume.ewm(span=12, adjust=False).mean()
            df_ad = pd.DataFrame(index=exp1.index,data={ 'acc_dist':''})
            for ind, row in self.df.iterrows():
                if row['High'] != row['Low']:
                    ac = ((row['Close'] - row['Low']) - (row['High'] - row['Close'])) / (row['High'] - row['Low']) * row['Volume']
                else:
                    ac = 0
                df_ad.at[ind, 'acc_dist']=ac
            df_ad=df_ad.cumsum()
            df_ad = pd.DataFrame(index=df_ad.index,data={'ad':df_ad.acc_dist.ewm(span=12, adjust=False).mean(),'close':self.df.Close.ewm(span=12, adjust=False).mean(),'g':''})

            df_ad['ad'] =df_ad['ad'].sub(df_ad['ad'].shift()).map(lambda x: np.sign(x))
            df_ad['close']=df_ad['close'].sub(df_ad['close'].shift()).map(lambda x: np.sign(x))
            df_ad['g']=df_ad[df_ad['ad']*df_ad['close']<0]
            df_ad[df_ad['g'].notnull()]
            df_ad['g'] = df_ad['g'][df_ad['g'].notnull()].map(lambda x: '1' if x == 1 else 'Z')
            # Z  : Accumulation Distribution decreases while price increases
            # 1 : Accumulation Distribution increases while price decreases

            self.df_final['ADL']= df_ad.g.values
            self.expressions['Z']="Accumulation Distribution decreases while price increases"
            self.expressions['1']= "Accumulation Distribution increases while price decreases"
            return self.df_final[self.df_final['ADL'].notnull()]

        def macd(self):
            exp1 = self.df.Close.ewm(span=12, adjust=False).mean()
            exp2 = self.df.Close.ewm(span=26, adjust=False).mean()
            macd = exp1-exp2
            exp3 = macd.ewm(span=9, adjust=False).mean()
            df_macd = pd.DataFrame({'Date':self.df.index, 'macd':macd.values,'sig':exp3.values})
            for i in range (exp1.index.size-1):
                if (macd[i]<exp3[i]) & (macd[i+1]>exp3[i+1]):
                    df_macd.loc[i+1,'trend']='O'
                elif (macd[i]>exp3[i]) & (macd[i+1]<exp3[i+1]):
                    df_macd.loc[i+1,'trend']='P'
                else: df_macd.loc[i+1,'trend'] =np.nan
            df_macd.loc[df_macd['trend']!='0',:]


            self.expressions['O']="MACD crossed above signal line"
            self.expressions['P']= "MACD crossed below signal line"
            self.df_final['MACD']= df_macd.trend.values
            return self.df_final[self.df_final['MACD'].notnull()]
        def acc(self):
            df_ad = pd.DataFrame(index=self.df.index,data={ 'acc_dist':''})
            for ind, row in self.df.iterrows():
                    if row['High'] != row['Low']:
                        ac = ((row['Close'] - row['Low']) - (row['High'] - row['Close'])) / (row['High'] - row['Low']) * row['Volume']
                    else:
                        ac = 0
                    df_ad.at[ind, 'acc_dist']=ac
            df_ad=df_ad.cumsum()
            df_ad = pd.DataFrame(index=df_ad.index,data={'ad':df_ad.acc_dist.ewm(span=12, adjust=False).mean(),'close':self.df.Close.ewm(span=12, adjust=False).mean(),'g':''})
            df_ad['ad'] =df_ad['ad'].sub(df_ad['ad'].shift()).map(lambda x: np.sign(x))
            df_ad['close']=df_ad['close'].sub(df_ad['close'].shift()).map(lambda x: np.sign(x))
            df_ad['g']=df_ad[df_ad['ad']*df_ad['close']<0]
            df_ad[df_ad['g'].notnull()]
            df_ad['g'] = df_ad['g'][df_ad['g'].notnull()].map(lambda x: '1' if x == 1 else 'Z')
            self.df_final['ADL']= df_ad.g.values
            return df_ad[df_ad['g'].notnull()]['g']

        def obv(self):

            obv = np.where(self.df['Close'] > self.df['Close'].shift(1), self.df['Volume'],
            np.where(self.df['Close'] < self.df['Close'].shift(1), -self.df['Volume'], 0)).cumsum()

            df_obv = pd.DataFrame(index=range(len(self.df.Close)),data={'obv':obv,'obvv':obv,'close':self.df.Close.values,'trend':''})
            df_obv['obv']=obv
            df_obv['obvv']=df_obv.obv.ewm(span=12, adjust=False).mean()
            df_obv['close']=df_obv.close.ewm(span=12, adjust=False).mean()
            df_obv['obvv'] =df_obv['obvv'].sub(df_obv['obvv'].shift()).map(lambda x: np.sign(x))
            df_obv['close']=df_obv['close'].sub(df_obv['close'].shift()).map(lambda x: np.sign(x))
            for i in range (df_obv.index.size-1):

                if (df_obv.loc[i,'obvv']*df_obv.loc[i,'close']) != (df_obv.loc[i+1,'obvv']*df_obv.loc[i+1,'close']):
                    if (df_obv.loc[i+1,'obvv'] == -1) & (df_obv.loc[i+1,'close']==1):
                        df_obv.loc[i+1,'trend']='S'
                    if (df_obv.loc[i+1,'obvv'] == 1) & (df_obv.loc[i+1,'close']==-1):
                        df_obv.loc[i+1,'trend']='T'
                    else: df_obv.loc[i+1,'trend'] =np.nan
                else: df_obv.loc[i+1,'trend'] =np.nan

            self.expressions['S']="OBV exponential moving average of 12 periods is decreasing while Price exponential moving average of 12 periods is increasing"
            self.expressions['T']= "OBV exponential moving average of 12 periods is increasing while Price exponential moving average of 12 periods is decreasing"
            self.df_final['OBV']= df_obv.trend.values
            return self.df_final[self.df_final['OBV'].notnull()]['OBV']

        def fin(self):
            self.df_final = self.df_final.replace(np.nan, '', regex=True)
            self.df_final = self.df_final.replace('0', '', regex=True)
            self.df_final['word']=self.df_final[self.df_final.columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
            self.df_final['word'] = self.df_final['word'].apply(lambda x : str(x))
            self.df_final['word'] =self.df_final['word'].apply(lambda x: '' if x=='nan' else x)
            return self.df_final['word']

if __name__ == "__main__":

    for filename in os.listdir(os.getcwd()+'/Data/SP500'):

        print(filename)
        if filename.split('.')[1]=='csv':
            sp = SP500_series(filename.split('.')[0])
            fn = filename.split('.')[0]
            a,b  = [5,20,50], [20,50,200]
            mac_sma = []
            mac_ema = []
            for i in range (2):
                mac_sma.append(sp.mac_sma(a[i],b[i]))
                mac_ema.append(sp.mac_sma(a[i],b[i]))
            rsi = sp.rsi()
            ao = sp.ao()
            pvo = sp.pvo()
            obv = sp.obv()
            acc = sp.acc()
            macd = sp.macd()
            ad = sp.ad()
            if os.path.exists(os.getcwd()+'/news-sp500/'+filename):
                sent = sp.sent()
            word = sp.fin()
            df = sp.df_final
            outdir_r = os.getcwd()+'/Results'
            if not os.path.exists(outdir_r):
                os.mkdir(outdir_r)
            outdir_r = os.getcwd()+'/Results/Quarter'
            if not os.path.exists(outdir_r):
                os.mkdir(outdir_r)
            outdir_r = os.getcwd()+'/Results/Year'
            if not os.path.exists(outdir_r):
                os.mkdir(outdir_r)
            outdir_r = os.getcwd()+'/Results/Month'
            if not os.path.exists(outdir_r):
                os.mkdir(outdir_r)
            outdir_r = os.getcwd()+'/Results/Total'
            if not os.path.exists(outdir_r):
                os.mkdir(outdir_r)





            outdir_df = os.getcwd()+'/Results/df'
            if not os.path.exists(outdir_df):
                os.mkdir(outdir_df)
            df.to_csv(os.path.join(outdir_df,filename))
            outdir_y = os.getcwd()+'/Results/Year/'+fn
            if not os.path.exists(outdir_y):
                os.mkdir(outdir_y)
            df['Y']=pd.PeriodIndex(df.index, freq='Y')
            for qt in df['Y'].unique():
                text_file = open(outdir_y+"/"+fn+'_'+str(qt)+".txt", "w")
                f_word = list(df[df['Y']==qt]['word'])
                text_file.write(fn+','+str(list(filter(lambda x: x!= '',f_word))))
                text_file.close()
            outdir_q = os.getcwd()+'/Results/Quarter/'+fn
            if not os.path.exists(outdir_q):
                os.mkdir(outdir_q)
            df['Q']=pd.PeriodIndex(df.index, freq='Q')
            for qt in df['Q'].unique():
                text_file = open(outdir_q+"/"+fn+'_'+str(qt)+".txt", "w")
                f_word = list(df[df['Q']==qt]['word'])
                text_file.write(fn+','+str(list(filter(lambda x: x!= '',f_word))))
                text_file.close()
            outdir_m = os.getcwd()+'/Results/Month/'+fn
            if not os.path.exists(outdir_m):
                os.mkdir(outdir_m)
            df['M']=pd.PeriodIndex(df.index, freq='M')
            for qt in df['M'].unique():
                text_file = open(outdir_m+"/"+fn+'_'+str(qt)+".txt", "w")
                f_word = list(df[df['M']==qt]['word'])
                text_file.write(fn+','+str(list(filter(lambda x: x!= '',f_word))))
                text_file.close()
            outdir_t = os.getcwd()+'/Results/Total'
            if not os.path.exists(outdir_t):
                os.mkdir(outdir_t)
            text_file = open(outdir_t+"/"+fn+".txt", "w")
            f_word = list(df['word'])
            text_file.write(fn+','+str(list(filter(lambda x: x!= '',f_word))))
            text_file.close()
            outdir_d = os.getcwd()+'/Results/dictionary'
            if not os.path.exists(outdir_d):
                    os.mkdir(outdir_d)
                    text_file = open(outdir_d+"/dictionary.txt", "w")
                    text_file.write(str(sp.expressions))
                    text_file.close()



