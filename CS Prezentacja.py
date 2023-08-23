#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center"><span style="font-size:56px"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Analiza i prognozowanie szereg&oacute;w czasowych <br />na przykładzie danych ze zbioru Bank Marketing UCI Machine Learning Repository </strong></span></span></p>

# <p style="text-align:center"></p>
# 
# <h2><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Cel prezentacji:</strong></span></h2>
# 
# <ul>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">przedstawienie proces&oacute;w podejmowania decyzji w trakcie analizy danych</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">zilustrowanie etapowania rozwoju modeli </span>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">uzględnienie biznesowych aspektów funkcjonowania przedsiębiorstwa do analizy </span>
# </ul>
# 
# 
# 
# <p></p>
# 

# <p style="text-align:center"></p>
# 
# <h2><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Opis problemu analitycznego:</strong><a class="anchor-link" href="http://localhost:8888/notebooks/Desktop/PSE/Prezentacja%20PSE.ipynb#Opis-problemu-analitycznego:">&para;&nbsp;</a></span></h2>
# 
# <ul>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"> zbiór danych to zestawienie informacji o kampanii marketingowej banku portugalskiego</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"> dane obejmują okres 2008-2010 przypadający na kryzys w strefie euro</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"> jako cel określono przewidywanie zgody klienta na założenie lokaty terminowej</span></li>
# </ul>
# 

# <p style="text-align:center">&nbsp;</p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h2><strong>Wykorzystane modele analityczne:</strong></h2>
# 
# <ol>
# 	<li>Regułowe - por&oacute;wnawcze</li>
# 	<li>XGBClassifier - eksplanacyjne </li>
# 	<li>AdsBoost - eksplanacyjne </li>
# 	<li>RandomForestClassifier - eksplanacyjne i predykcyjne </li>
# </ol>
# </span>
# <p>&nbsp;</p>
# 

# <p style="text-align:center">&nbsp;</p>
# 
# <p><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Wykorzystane narzędzia w projekcie:</strong></span></p>
# 
# <ol>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">IDE oraz prezentacji<strong>:</strong></span><span style="font-family:Courier New,Courier,monospace"> jupyter notebook, nbextensions, RISE</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Podstawowa manipulacja danymi: </span><span style="font-family:Courier New,Courier,monospace">sys, os, pandas, numpy, calendar</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Redukcja wymiarowości i próbkowanie niezbalansowanych zbiorów: </span><span style="font-family:Courier New,Courier,monospace">prince, imblearn </span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Modele eksplanacyjne i określające dyskryminację: </span><span style="font-family:Courier New,Courier,monospace">dalex</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Model ML i metryki błędu: </span><span style="font-family:Courier New,Courier,monospace">sklearn, xgboost</span></li>
# 	<li><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">Narzędzia graficzne: </span><span style="font-family:Courier New,Courier,monospace">seaborn, matplotlib </span><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">, </span><span style="font-family:Courier New,Courier,monospace">matplotlib</span></li>
# 	
# </ol>
# 
# <p>&nbsp;</p>
# 

# In[1]:


#import pakietów 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import calendar
import datetime
import seaborn as sns
import dalex as dx
import xgboost as xgb
import imblearn
import warnings
import prince 



# In[2]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#wczytanie pliku i podsumowanie
df = pd.read_csv('bank-additional-full.csv', delimiter = ';')
df.info()
df.shape[0]


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Wstępna obr&oacute;bka danych </strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Problemy</strong></span></h4>
# <ul>
# <li>Dane posiadają braki rekordów w postaci "unknown" dla 6 różnych zmiennych o różnej częstości</li>
# <li>Dane nie są indeksowane czasowo, pomimo zachowania ich kolejności zbiorze</li>
# <li>Pewne zmienne zawierają rekordy specjalne oznaczające wydarzenie np. '999' w 'pdays' oznacza brak kontaktu, w odróżnieniu od dni od ostatniego kontaktu z klientem jak w przypadku pozostałych rekordów </li>
# </ul>
# 
# 
# 

# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Wstępna obr&oacute;bka danych </strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Rozwiązania</strong></span></h4>
# <ul>
# <li>Dodano zmienne binarne referncyjne dla danych zmiennych zawierających 'unknown' aby model mógł odóżnić je od zwykłej klasy kategorialnej</li>
# <li>Wyprowadzono indeksakcje dni na podstawie danych z dokumentacji oraz przełomów miesiąca widocznych w następujących rekordach zmiennej 'month'</li>
# <li>Analogicznie dodano zmienne binarne referncyjne dla danych zmiennych zawierających rekordy specjalne </li>
# </ul>
# 
# 

# In[4]:


#*****************************************************************************************************************************
#Start EDA
#*****************************************************************************************************************************


# In[5]:


#Identyfikacji wartości brakujących z dokumentacji -  8. Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques. 
unknown_values=['unknown']
df.isin(unknown_values).sum()


# In[6]:


#Kolumny oznaczające brakującą wartość w zależności od typu
columns_with_unknown_values=['job','marital','education','default','housing','loan']
missing_data=[]
for j in range(0,len(columns_with_unknown_values)):
    col_name = 'Missing_' + str(columns_with_unknown_values[j])
    df.loc[:, col_name] = df[columns_with_unknown_values[j]].isin(unknown_values)
    missing_data.append(col_name)


# In[7]:


missing_data


# In[8]:


#Podusmowanie brakujących danych procentowo
columns_with_unknown_values_rank=(df[['job','marital','education','default','housing','loan']].isin(unknown_values).sum()/df.shape[0]).sort_values(ascending=False)
columns_with_unknown_values_rank


# In[9]:


#zmienna zależna:
target_variable=['y']
#Zestawienie kolumn zgodnie z opisem dokumentacji 
client_data_columns=['age','job','marital','education','default','housing','loan','contact']
campaing_data_columns=['contact','month','day_of_week','duration','campaign','pdays','previous','poutcome']
context_data_columns=['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
#Zestawienie kolumn zgodnie z interwałem czasowym
time_quarterly_data=['emp.var.rate','nr.employed']
time_monthly_data=['cons.price.idx','cons.conf.idx']
time_daily_data=['euribor3m','day_of_week']
#Zestawienie kolumn sterowalnych: decyzja o ilości zatrudnionych
managed_data_columns=['nr.employed']
#Zestawienie kolumn zabronionych do analizy predykcyjnej
forbidden_columns=['duration','y'] # Z dokuemntacji: Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#Zestawienie kolumn numerycznych
numeric_columns=['age', 'campaign','pdays','previous','emp.var.rate','nr.employed','cons.price.idx','cons.conf.idx','euribor3m']
#Zestawienie kolumn kategorialnych
categorical_columns=['job','marital','education','default','housing','loan','month','day_of_week','poutcome','contact']


# In[10]:


#Oznaczenie klientów, którzy byli kontaktowani podczas wcześniejszej kampani 
df['was_contacted']=df['pdays'].apply(lambda x: False if x==999 else True)
categorical_columns.append('was_contacted')


# In[11]:


#Oznaczenie klientów, którzy są kontaktowani po raz pierwszy podczas akrualnej kampanii 
df['first_contacted']=df['previous'].apply(lambda x: True if x==0 else False)
categorical_columns.append('first_contacted')


# In[12]:


#Oznaczenie klientów, którzy są kontaktowani po raz pierwszy i jedyny podczas aktualnej kampanii 
def _once_contacted(row):    
    if row["first_contacted"] == True:
        if row ['campaign'] == 1: 
            value=True  
        else: value=False 
    else: value=False 
    return value
df['once_contacted'] = df.apply(_once_contacted, axis=1)
categorical_columns.append('once_contacted')


# In[13]:


#Wyznaczenie roku dla każdego rekordu, ze względu wskazówki z dokumentacji: 1) bank-additional-full.csv with all examples, ordered by date (from May 2008 to November 2010).
years_arr=[]
year=2008
for i in range(df.shape[0]):
    if (df.iloc[i]['month']!='dec' and df.iloc[i-1]['month']=='dec'):
        year=year+1
        years_arr.append(year)
    else:    
        years_arr.append(year)
df['year'] = pd.DataFrame(years_arr)
categorical_columns.append('year')


# In[14]:


#Wyznaczenie kolejnego dnia pracującego w miesiącu na podstawie dni tygodnia i miesięcy - nie pozwala to ustalić dokładnie kalendarzowej daty
days_arr=[]
cumulative_arr=[]
working_day=1
cumulative=1
for i in range(df.shape[0]):
    if (df.iloc[i]['month']!=df.iloc[i-1]['month']):
        working_day=1
        days_arr.append(working_day)
        cumulative=cumulative+1
        cumulative_arr.append(cumulative)
    elif(df.iloc[i]['day_of_week']=="mon" and df.iloc[i-1]['day_of_week']=='fri'):   
        working_day=working_day+2
        days_arr.append(working_day)
        cumulative=cumulative+1
        cumulative_arr.append(cumulative)
    elif(df.iloc[i]['day_of_week']!=df.iloc[i-1]['day_of_week']):
        working_day=working_day+1
        days_arr.append(working_day)
        cumulative=cumulative+1
        cumulative_arr.append(cumulative)
    else:
        days_arr.append(working_day)
        cumulative_arr.append(cumulative)

df['day'] = pd.DataFrame(days_arr)
df['cumulative_days'] = pd.DataFrame(cumulative_arr)
categorical_columns.append('day')
numeric_columns.append('cumulative_days')


# 
# 
# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Wstęp</strong></span></h4>
# <ul>
# <li>Wstępnie dodatkowo wyprowadzono dodatkowe zmienne wskaźnikowe dla zmiennych ilościowych zmiennych w czasie odniesione do dnia wcześniejszego *'_ind' i pierwszego dnia zbiorze danych *'_ind_l'</li>
# <li>Wyprowadzono zmienne dni skumulowanych 'cumulative_days' oraz krzywej uczenia orgnizacji 'learning_curve' obrazującą doświadczenie zespołu marketingowego </li>
# <li>Sprawdzono różnicę zmiennej zależnej w stosunku do dnia tygodnia</li>
# </ul>
# 
# 
# 
# 

# In[15]:


#zamiana skrótów nazw miesiąca na numery 
from calendar import month_abbr
lower_ma = [m.lower() for m in month_abbr]
df['month_number'] = df['month'].str.lower().map(lambda m: lower_ma.index(m)).astype('Int8')
numeric_columns.append('month_number')
#pełna data przybliżona
df['date'] = pd.to_datetime(dict(year=df.year, month=df.month_number, day=df.day))
numeric_columns.append('date')
#numer dnia tygodnia
days = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}
df['day_of_week_nr']= df['day_of_week'].apply(lambda x: days[x])
numeric_columns.append('day_of_week_nr')


# In[16]:


#wskaźnik dynamiki od ostatniego pomiaru euribor3m - dniowy
df_euro=df.groupby(['year','month_number','day'])['euribor3m'].mean().to_frame()
df_euro['euribor3m_ind']=(df_euro['euribor3m'].shift(1))/(df_euro['euribor3m'].shift(2))
df_euro=df_euro.fillna(1)
df_euro=df_euro.reset_index()
df_euro=df_euro.drop(columns=['euribor3m'])
df=df.merge(df_euro, on=['year','month_number','day'])
numeric_columns.append('euribor3m_ind')



# In[17]:


#wskaźnik dynamiki od piewszego pomiaru euribor3m - dniowy
df_euro_l=df.groupby(['year','month_number','day'])['euribor3m'].mean().to_frame()
df_euro_l=df_euro_l.reset_index()
df_euro_l['euribor3m_ind_l']=(df_euro_l['euribor3m'])/(df_euro_l['euribor3m'].iloc[0])
df_euro_l=df_euro_l.drop(columns=['euribor3m'])
df_euro_l
df=df.merge(df_euro_l, on=['year','month_number','day'])
numeric_columns.append('euribor3m_ind_l')


# In[18]:


#wskaźnik dynamiki ostatniego pomiaru cons.price.idx - miesięczny
df_cons=df.groupby(['year','month_number'])['cons.price.idx'].mean().to_frame()
df_cons['cons_price_ind']=(df_cons['cons.price.idx'].shift(1))/(df_cons['cons.price.idx'].shift(2))
df_cons=df_cons.fillna(1)
df_cons=df_cons.reset_index()
df_cons=df_cons.drop(columns=['cons.price.idx'])
df=df.merge(df_cons, on=['year','month_number'])
numeric_columns.append('cons_price_ind')




# In[19]:


#wskaźnik dynamiki od piewszego pomiaru cons.price.idx - miesięczny
df_cons_l=df.groupby(['year','month_number','day'])['cons.price.idx'].mean().to_frame()
df_cons_l=df_cons_l.reset_index()
df_cons_l['cons_price_ind_l']=(df_cons_l['cons.price.idx'])/(df_cons_l['cons.price.idx'].iloc[0])
df_cons_l=df_cons_l.drop(columns=['cons.price.idx'])
df_cons_l
df=df.merge(df_cons_l, on=['year','month_number','day'])
numeric_columns.append('cons_price_ind_l')


# In[20]:


#wskaźnik dynamiki ostatniego pomiaru cons_conf.conf.idx - miesięczny
df_cons_conf=df.groupby(['year','month_number'])['cons.conf.idx'].mean().to_frame()
df_cons_conf['cons_conf_ind']=(df_cons_conf['cons.conf.idx'].shift(1))/(df_cons_conf['cons.conf.idx'].shift(2))
df_cons_conf=df_cons_conf.fillna(1)
df_cons_conf=df_cons_conf.reset_index()
df_cons_conf=df_cons_conf.drop(columns=['cons.conf.idx'])
df=df.merge(df_cons_conf, on=['year','month_number'])
numeric_columns.append('cons_conf_ind')


# In[21]:


#wskaźnik dynamiki od piewszego pomiaru cons.conf.idx- miesięczny
df_cons_conf_l=df.groupby(['year','month_number'])['cons.conf.idx'].mean().to_frame()
df_cons_conf_l=df_cons_conf_l.reset_index()
df_cons_conf_l['cons_conf_ind_l']=(df_cons_conf_l['cons.conf.idx'])/(df_cons_conf_l['cons.conf.idx'].iloc[0])
df_cons_conf_l=df_cons_conf_l.drop(columns=['cons.conf.idx'])
df=df.merge(df_cons_conf_l, on=['year','month_number'])
numeric_columns.append('cons_conf_ind_l')


# In[22]:


#wskaźnik dynamiki  od ostaniego pomiaru  emp.var.rate - miesięczny,dzielony przez wartość bezwzględną, ze względu na wartości ujemne
df_emp_var_ind=df.groupby(['year','month_number'])['emp.var.rate'].mean().to_frame()
df_emp_var_ind['emp_var_ind']=(df_emp_var_ind['emp.var.rate'].shift(1))/abs((df_emp_var_ind['emp.var.rate'].shift(2)))
df_emp_var_ind=df_emp_var_ind.fillna(1)
df_emp_var_ind=df_emp_var_ind.reset_index()                             
df_emp_var_ind=df_emp_var_ind.drop(columns=['emp.var.rate'])
df=df.merge(df_emp_var_ind, on=['year','month_number'])
numeric_columns.append('emp_var_ind')


# In[23]:


#wskaźnik dynamiki od piewszego pomiaru emp.var.rate - miesięczny,dzielony przez wartość bezwzględną, ze względu na wartości ujemne
df_emp_var_ind_l=df.groupby(['year','month_number'])['emp.var.rate'].mean().to_frame()
df_emp_var_ind_l=df_emp_var_ind_l.reset_index()
df_emp_var_ind_l['emp_var_ind_l']=(df_emp_var_ind_l['emp.var.rate'])/abs((df_emp_var_ind_l['emp.var.rate'].iloc[0]))
df_emp_var_ind_l=df_emp_var_ind_l.drop(columns=['emp.var.rate'])
df=df.merge(df_emp_var_ind_l, on=['year','month_number'])
numeric_columns.append('emp_var_ind_l')


# In[24]:


#wskaźnik dynamiki od piewszego pomiaru nr.employed- miesięczny
df_emp_nr_ind=df.groupby(['year','month_number'])['nr.employed'].mean().to_frame()
df_emp_nr_ind['emp_nr_ind']=(df_emp_nr_ind['nr.employed'])/(df_emp_nr_ind['nr.employed'].iloc[0])
df_emp_nr_ind=df_emp_nr_ind.reset_index()
df_emp_nr_ind=df_emp_nr_ind.drop(columns=['nr.employed'])
df=df.merge(df_emp_nr_ind, on=['year','month_number'])
numeric_columns.append('emp_nr_ind')


# In[25]:


#wskaźnik dynamiki ostatniego pomiaru nr.employed- miesięczny
df_emp_nr_ind_l=df.groupby(['year','month_number'])['nr.employed'].mean().to_frame()
df_emp_nr_ind_l['emp_nr_ind_l']=(df_emp_nr_ind_l['nr.employed'])/(df_emp_nr_ind_l['nr.employed'].shift(2))
df_emp_nr_ind_l=df_emp_nr_ind_l.fillna(1)
df_emp_nr_ind_l=df_emp_nr_ind_l.reset_index()
df_emp_nr_ind_l=df_emp_nr_ind_l.drop(columns=['nr.employed'])
df=df.merge(df_emp_nr_ind_l, on=['year','month_number'])
numeric_columns.append('emp_nr_ind_l')


# In[26]:


#krzywa uczenia dla organizacji 
df['learning_curve'] = ((df['nr.employed']*df['cumulative_days'])**0.5)
numeric_columns.append('learning_curve')


# In[27]:


#wskaźnik sukcesu dziennego 
#zmienna zależna jako zmienna numeryczna
df['numeric_y']=df['y'].apply(lambda x: 1 if x=='yes' else 0)
numeric_columns.append('numeric_y')

df_y_nr=df.groupby(['year','month_number','day'])['numeric_y'].mean().to_frame()
df_y_nr['daily_success_rate']=df_y_nr['numeric_y']
df_y_nr=df_y_nr.reset_index()
df_y_nr=df_y_nr.drop(columns=['numeric_y'])
df=df.merge(df_y_nr, on=['year','month_number','day'])



# In[28]:


#średni wskaźnik czasu rozmowy dla sukcesu 
df_dur = df.replace(0, np.NaN)
df_dur=df.groupby(['year','month_number','day'])['duration'].mean().to_frame()
df_dur['success_mean_time']=df_dur['duration']
df_dur=df_dur.reset_index()
df_dur=df_dur.drop(columns=['duration'])
df=df.merge(df_dur, on=['year','month_number','day'])



# In[29]:


#suma sukcesów dziennych 
df_y_sum=df.groupby(['year','month_number','day'])['numeric_y'].sum().to_frame()
df_y_sum['daily_success_sum']=df_y_sum['numeric_y']
df_y_sum=df_y_sum.reset_index()
df_y_sum=df_y_sum.drop(columns=['numeric_y'])
df=df.merge(df_y_sum, on=['year','month_number','day'])



# In[30]:


#suma kontaktów dziennych
df_contacts=df.groupby(['year','month_number','day'])['numeric_y'].count().to_frame()
df_contacts['daily_contacts_sum']=df_contacts['numeric_y']
df_contacts=df_contacts.reset_index()
df_contacts=df_contacts.drop(columns=['numeric_y'])
df=df.merge(df_contacts, on=['year','month_number','day'])



# In[31]:


#wskaźnik kontaktów  na pracownika
df_er_sum=df.groupby(['year','month_number','day'])['numeric_y'].count().to_frame()
df_temp=df.groupby(['year','month_number','day'])['nr.employed'].mean().to_frame()
df_er_sum['emp_eff_rate']=(df_er_sum['numeric_y'])/(df_temp['nr.employed'])
df_er_sum=df_er_sum.reset_index()
df_er_sum=df_er_sum.drop(columns=['numeric_y'])
df=df.merge(df_er_sum, on=['year','month_number','day'])



# In[32]:


#wskaźnik dynamiki kontaktów  na pracownika od ostaniego pomiaru  emp_eff_rate - dzienny

df_emp_eff_ind=df.groupby(['year','month_number','day'])['emp_eff_rate'].mean().to_frame()                                                         
df_emp_eff_ind['emp_eff_ind']=(df_emp_eff_ind['emp_eff_rate'].shift(1))/(df_emp_eff_ind['emp_eff_rate'].shift(2))
df_emp_eff_ind=df_emp_eff_ind.fillna(1)
df_emp_eff_ind=df_emp_eff_ind.reset_index()                             
df_emp_eff_ind=df_emp_eff_ind.drop(columns=['emp_eff_rate'])
df=df.merge(df_emp_eff_ind, on=['year','month_number','day'])
numeric_columns.append('emp_eff_ind')


# In[33]:


#wskaźnik dynamiki kontaktów  na pracownika od początku pomiaru emp_eff_rate - dzienny
df_emp_eff_var=df.groupby(['year','month_number','day'])['emp_eff_rate'].mean().to_frame()
df_emp_eff_var=df_emp_eff_var.reset_index()                                                     
df_emp_eff_var['emp_eff_ind_l']=(df_emp_eff_var['emp_eff_rate'])/(df_emp_eff_var['emp_eff_rate'].iloc[0])
df_emp_eff_var=df_emp_eff_var.fillna(1)
df_emp_eff_var=df_emp_eff_var.reset_index()                             
df_emp_eff_var=df_emp_eff_var.drop(columns=['emp_eff_rate'])
df=df.merge(df_emp_eff_var, on=['year','month_number','day'])
numeric_columns.append('emp_eff_ind_l')


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Ustalenia - zmienne ilościowe w funkcji czasu</strong></span></h4>
# <ul>
# <li>Spadek euriboru koreluje ze wzrostem wskaźnika sukcesu, zauważalana zmiana od początku 2009</li>
# <li>Krzywa uczenia koreluje ze wskaźnikiem dzinnego sukcesu </li>
# <li>Brak jednoznacznej korelacji średniego dzienny czasu rozmowy z klientem i wskaźnika dziennego sukcesu</li>
# <li>Poniedziałki są najsłabsze w każdym roku i przeciętnie piątki również poniżej średniej  </li>
# <li>Wskaźnik sukcesów rośnie wraz ze spadkiem ilości sukcesów - większa efektywność ale nie ilość sukcesów, zauważalana zmiana od początku 2009 </li>
# <li>Wskaźnik sukcesów utrzymuje się na podobnym poziomie przy zmniejszającej się liczbie pracowników - większa efektywność ale nie ilośc sukcesów, zauważalana zmiana od początku 2009  </li>
# </ul>
# 

# In[34]:


#wykres w funkcji czasu - euribor vs. wskaźnik dzinnego sukcesu
ax = df.plot(x="date", y="euribor3m", legend=False)
ax.set_ylabel('Euribor 3 month rate')
ax2 = ax.twinx()
ax2.set_ylabel('Daily success rate')
df.plot(x="date", y="daily_success_rate", ax=ax2, legend=False, color="r")

ax.figure.legend(title='Legend:', loc='upper right', labels = ['Euribor 3 month rate - daily indicator','Daily success rate']) 
plt.show()

#wykres w funkcji czasu - spadek euriboru koreluje ze wzrostem wskaźnika sukcesu, zauważalana zmiana od początku 2009


# In[35]:


#wykres w funkcji czasu - krzywa uczenia się vs. wskaźnik dzinnego sukcesu
ax = df.plot(x="date", y="learning_curve", legend=False, color="black")
ax.set_ylabel('Learning curve')
ax2 = ax.twinx()
ax2.set_ylabel('Daily success rate')
df.plot(x="date", y="daily_success_rate", ax=ax2, legend=False, color="r")

ax.figure.legend(title='Legend:', loc='upper right', labels = ['Learning curve - employee number times cumulative wroking days ','Daily success rate']) 
plt.show()
# krzywa uczenia koreluje ze wskaźnikiem dzinnego sukcesu


# In[36]:


#wykres w funkcji czasu -średni dzienny czas rozmowy z kleintem się vs. wskaźnik dzinnego sukcesu
ax = df.plot(x="date", y='success_mean_time', legend=False, color="cyan")
ax.set_ylabel('Duration time of succeded contact mean')
ax2 = ax.twinx()
ax2.set_ylabel('Daily success rate')
df.plot(x="date", y="daily_success_rate", ax=ax2, legend=False, color="r")

ax.figure.legend(title='Legend:', loc='upper right', labels = ['Duration time of succeded contact - daily mean','Daily success rate']) 
plt.show()
# brak jednoznacznej korelacji


# In[37]:


#Sprawdzenie rozkłądów dla danych dni tygodnia w danym roku
df_dow_nr=df.groupby(['year','day_of_week_nr'])['numeric_y'].mean().to_frame()
df_dow_nr['day_of_week_success_rate']=df_dow_nr['numeric_y']
df_dow_nr=df_dow_nr.drop(columns=['numeric_y'])
df_dow_nr
#poniedziałki najsłabsze w każdym roku, 


# In[38]:


#Sprawdzenie rozkłądów dla danych dni tygodnia w całym okresie
df_dow_nr2=df.groupby(['day_of_week_nr'])['numeric_y'].mean().to_frame()
df_dow_nr2['day_of_week_success_rate_mean']=df_dow_nr2['numeric_y']/df_dow_nr2['numeric_y'].mean()
df_dow_nr2=df_dow_nr2.drop(columns=['numeric_y'])
df_dow_nr2
# średnio piątki również 


# In[39]:


#wykres w funkcji czasu -ilość sukcesów do wskaźnika sukcesów
ax = df.plot(x="date", y='daily_success_sum', legend=False,color="yellow")
ax.set_ylabel('Daily success sum')
ax2 = ax.twinx()
ax2.set_ylabel('Daily success rate')
df.plot(x="date", y="daily_success_rate", ax=ax2, legend=False, color="r")

ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily success sum ','Daily success rate - daily indicator']) 
plt.show()
# wskaźnik sukcesów rośnie wraz ze wpadkiem ilości sukcesów - większa efektywność ale nie ilośc sukcesów, zauważalana zmiana od początku 2009


# In[40]:


#wykres w funkcji czasu -ilość pracowników vs. wskaźnik sukcesu
ax.set_ylabel('Daily success rate')
ax = df.plot(x="date", y="daily_success_rate", legend=False, color="r")
ax2 = ax.twinx()
ax2.set_ylabel('Number of employees')
df.plot(x="date", y="nr.employed", ax=ax2, legend=False, color="g")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily success rate - daily indicator','Number of employees - quarterly indicator']) 
plt.show()
# wskaźnik sukcesów utrzymuje się na podobnym poziomie przy zmniejszonej liczbie praconików  - większa efektywność ale nie ilośc sukcesów, zauważalana zmiana od początku 2009


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Ustalenia - zmienne ilościowe w funkcji czasu cd.</strong></span></h4>
# <ul>
# <li>Wskaźnik sukcesów utrzymuje się na podobnym poziomie przy zmniejszonej liczbie kontaktów - większa efektywność ale nie ilość sukcesów, zauważalana zmiana od początku 2009</li>
# <li>Wskaźnik kontaktów utrzymuje się na podobnym poziomie przy zmniejszającej się liczbie pracowników - zauważalana zmiana od połowy 2009, gdzie spadki zaczynają korelelować </li>
# <li>Ilość sukcesów utrzymuje się na podobnym poziomie przy zmniejszającej się liczbie pracowników - zauważalny jednorazowy skok na początku 2009 roku </li>
# <li>Ilość kontaktów utrzymuje się na podobnym poziomie przy zmniejszającej się liczbie pracowników - zauważalna zmania od połowy 2009 roku   </li>
# <li>Silna korelacja wskażnika euribor z liczbą pracowników z niewielką zmianą trendu pod koniec szeregu czasowego analizowanych danych</li>
# </ul>
# 

# In[41]:


#wykres w funkcji czasu -wskażnik kontaktów z klientem vs. wskaźnik sukcesu
ax.set_ylabel('Daily success rate')
ax = df.plot(x="date", y="daily_success_rate", legend=False, color="r")
ax2 = ax.twinx()
ax2.set_ylabel('Daily contact rate')
df.plot(x="date", y='emp_eff_rate',ax=ax2, legend=False, color="violet")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily success rate - daily indicator','Daily contact rate - daily indicator']) 
plt.show()
# wskaźnik sukcesów utrzymuje się na podobnym poziomie przy zmniejszonej liczbie kontaktów  - większa efektywność ale nie ilośc sukcesów, zauważalana zmiana od początku 2009


# In[42]:


#wykres w funkcji czasu -wskażnik kontaktów z klientem vs. liczba pracowników
ax.set_ylabel('Daily contact rate')
ax = df.plot(x="date", y='emp_eff_rate', legend=False, color="violet")
ax2 = ax.twinx()
ax2.set_ylabel('Number of employees')
df.plot(x="date", y="nr.employed", ax=ax2, legend=False, color="g")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily contact rate - daily indicator','Number of employees - quarterly indicator']) 
plt.show()
# wskaźnik konaktów utrzymuje się na podobnym poziomie przy zmniejszonającej się liczbie pracowników  - wizauważalana zmiana od połowy 2009, gdzie spadki zaczynają korelelować


# In[43]:


#wykres w funkcji czasu - ilość sukcesów vs. liczba pracowników
ax = df.plot(x="date", y='daily_success_sum', legend=False,color="yellow")
ax.set_ylabel('Daily success sum')
ax2 = ax.twinx()
ax2.set_ylabel('Number of employees')
df.plot(x="date", y="nr.employed", ax=ax2, legend=False, color="g")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily success sum','Number of employees - quarterly indicator']) 
plt.show()
# ilość sukcesów utrzymuje się na podobnym poziomie przy zmniejszonającej się liczbie pracowników  -zauważalny  jednorazowy skok na początku 2009 roku 


# In[44]:


#wykres w funkcji czasu - ilość kontaktów vs. liczba pracowników
ax = df.plot(x="date", y='daily_contacts_sum', legend=False,color="brown")
ax.set_ylabel('Daily contacts sum')
ax2 = ax.twinx()
ax2.set_ylabel('Number of employees')
df.plot(x="date", y="nr.employed", ax=ax2, legend=False, color="g")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily contacts sum','Number of employees - quarterly indicator']) 
plt.show()
# ilość kontaktów utrzymuje się na podobnym poziomie przy zmniejszonającej się liczbie pracowników  -zauważala zmania od połowy 2009 roku 


# In[45]:


#wykres w funkcji czasu - wskażniki euribor vs. liczba pracowników
ax = df.plot(x="date", y="euribor3m", legend=False)
ax.set_ylabel('Euribor 3 month rate')
ax2 = ax.twinx()
ax2.set_ylabel('Number of employees')
df.plot(x="date", y="nr.employed", ax=ax2, legend=False, color="g")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Euribor 3 month rate - daily indicator','Number of employees - quarterly indicator']) 
plt.show()
# silna korelacja wskażnika euribor z. liczba pracowników z  niewielką zmianą terndu pod koniec szeregu czasowego analizwoanych danych


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Ustalenia - zmienne ilościowe w funkcji czasu cd.</strong></span></h4>
# <ul>
# <li>Silna korelacja wskaźnika wskażnika euribor ze wskaźnikiem rotacji zatrudnienienia ze zmianą trendu od początku 2010 roku</li>
# <li>Silna korelacja wskaźnika euribor ze wskanikiem cen ze zmianą trendu od połowy 2009 </li>
# <li>Silna korelacja wskaźnika cen ze wskaźnikiem rotacji zatrudnienienia pozostająca pomimo odwrócenia trendu w połowie 2009</li>
# <li>Przy spadku wskaźnika cen utrzymuje się podobna ilość kontaktów do połowy 2009, potem wraz ze wzrostem wskaźnika cen spada liczba kontaktów</li>
# <li>Dokonano kodowania zmiennych kategorialnych na porządkowe ('factorized_*') do zasostowania macierzy korelacji ze zmiennymi ilościowymi</li>
# </ul>
# 

# In[47]:


#wykres w funkcji czasu - wskażniki euribor vs. wskaźnik rotacji zatrudnienienia 
ax = df.plot(x="date", y="euribor3m", legend=False)
ax.set_ylabel('Euribor 3 month rate')
ax2 = ax.twinx()
ax2.set_ylabel('Employment variation rate')
df.plot(x="date", y="emp.var.rate", ax=ax2, legend=False, color="magenta")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Euribor 3 month rate - daily indicator','Employment variation rate- quarterly indicator']) 
plt.show()
# silna korelacja wskażnika wskażniki euribor ze wskaźnikiem rotacji zatrudnienienia  ze zmianę trendu od początku 2010


# In[48]:


#wykres w funkcji czasu - wskażniki euribor vs. wskaźnik cen
ax = df.plot(x="date", y="euribor3m", legend=False)
ax.set_ylabel('Euribor 3 month rate')
ax2 = ax.twinx()
ax2.set_ylabel('Consumer price index')
df.plot(x="date", y="cons.price.idx", ax=ax2, legend=False, color="orange")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Euribor 3 month rate - daily indicator','Consumer price index - monthly indicator']) 
plt.show()
# silna korelacja wskażnika wskażniki euribor ze cen  ze zmianę trendu od połowy 2009


# In[49]:


#wykres w funkcji czasu -  wskaźnik cen  vs. wskaźnik rotacji zatrudnienienia 
ax = df.plot(x="date", y="cons.price.idx", legend=False, color="orange")
ax.set_ylabel('Consumer price index')
ax2 = ax.twinx()
ax2.set_ylabel('Employment variation rate')
df.plot(x="date", y="emp.var.rate", ax=ax2, legend=False, color="magenta")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Consumer price index - monthly indicator','Employment variation rate- quarterly indicator']) 
plt.show()
# silna korelacja  wskaźnik cen ze wskaźnikiem rotacji zatrudnienienia - korelacja pozostaje przy zmianej trendu w połowie 2009


# In[50]:


#wykres w funkcji czasu -  wskaźnik cen  vs. liczba kontatków z kielntem 
ax = df.plot(x="date", y='daily_contacts_sum', legend=False,color="brown")
ax.set_ylabel('Daily contacts sum')
ax2 = ax.twinx()
ax2.set_ylabel('Consumer price index')
df.plot(x="date", y="cons.price.idx", ax=ax2, legend=False, color="orange")
ax.figure.legend(title='Legend:', loc='upper right', labels = ['Daily contacts sum','Consumer price index - monthly indicator']) 
plt.show()
# przy spadku wskaźnika cen utrzymuje się podobna ilość kontaktów do połowy 2009, potem wraz ze wzrostem wskaźnika cen spada liczba kontaktów


# In[51]:


#kodowanie kolumn kategorialnych 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
factorized_columns=[]
to_label_columns=categorical_columns+missing_data
for i in to_label_columns:
        df['factorized_'+i] = le.fit_transform(df[i])
        factorized_columns.append('factorized_'+i)
#kolumny do analizy korelacji  
correlation_columns=numeric_columns+factorized_columns


# In[52]:


#Wykres korelacji
plt.figure(figsize=(16, 10))
heatmap = sns.heatmap(df[correlation_columns].corr()[['numeric_y']].sort_values(by='numeric_y', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features correlating with numeric_y', fontdict={'fontsize':18}, pad=16);


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Podsumowanie - zmienne ilościowe w funkcji czasu</strong></span></h4>
# <ul>
# <li>Potencjalnie głównie czynnikiem wpływającym na zmienną zależną dodatnio jest sam czas reprezentowany przez inne zmienne </li>
# <li>Czynnkiem wpływającym na zmienną zależną negatywnie są różnice w rynku pracy, wskaźnik zatrudnienia pracowników oraz euribor</li>
# <li>Efektywność kampanii nie wiązała się bezpośrednio z ilością pracowników</li>
# <li>Efektywność kampanii wiązała się z silnym spadkiem euriboru, nabyciem doświadczenia przez kadrę po pierwszym roku</li>
# <li>Pozostałe wskaźniki ekonomiczne korelują pozytywnie od połowy szeregu potem negetywnie z efektywnością kampanii</li>
# </ul

# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - wstęp</strong></span></h4>
# <ul>
# <li>Zmienna zależna nominalna ma rozkład silnie niesymetryczny - 89% wartości przyjmuje wartość negatywną  </li>
# <li>Skutkuje to potencjalnymi problemami z odpowiednim próbkowaniem zbioru i  dyskryminacją osób ze względu na niską reprezentacją danej zmiennej w zbiorze </li>
# <li>Jako wartość refencyjną do oceny modeli bez równoważania rozkładu zmiennej zależnej metodami próbkowania przyjęto 89% dla metryki dokładność ('accuracy') </li>
# 
# </ul>

# In[53]:


fig, ax = plt.subplots(figsize=(12, 3))
#jaki rozkład ma zmienna zależna)
sns.countplot(x='y', data=df, ax=ax)
plt.show()
df.y.value_counts()
#silnie niesymetryczny  


# In[54]:


#wynik referencyjny dla modelu uczenia maszynowaego bez użycia technik balansujących 
current_campaign_result=df.y.value_counts()[1]/df.shape[0]
current_campaign_result


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - wstęp cd.</strong></span></h4>
# <ul>
# <li>Zmienną wieku zdyskretyzowano kwantylowo celemzbalansowania i wydzielenia adekwatnych grup porównawczych do segmentacji klientów</li>
# <li>Rozkład kategorii w zmiennej zawodu jest silnie niesymetryczny - 4 kategorie zawodów z 12 powyżej średniej  </li>
# <li>Kategorię 'housemaid' w tej zmiennej wyróżniono jako potencjalnie silnie dyskryminujacą jako zawierającą informację o płci i nieproporcjonalnie reprezentowaną w zbiorze</li>
# 
# </ul>

# In[55]:


# dyskretyzacja kwantylowa dla wieku celem ustalenia równych przediałów w uwagi na częśtość występowania danej kategorii - przyjęto liczbe przedziłów równą ilości kategorii zmiennej o jej większej wartości  
qc= pd.qcut(df['age'], q=df['job'].nunique(), precision=0)
qc=qc.to_frame()
qc=qc.rename(columns={"age": "age_bin"})
df=df.merge(qc, left_index=True,right_index=True )


# In[56]:


#jaki rozkład ma zmienna wieku 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='age_bin', data=df, ax=ax,hue="y")
plt.show()
df.age_bin.value_counts()
#stosunkowo dobrze zabalnsowana po transformacji, 


# In[57]:


#jaki rozkład ma zmienna zawodu
sns.set(font_scale = 0.7)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='job', data=df, ax=ax,hue="y")
plt.show()
print(df.job.value_counts())
print('Categories no.: '+ str(df.job.nunique()))
print('Mean: '+ str(df.job.value_counts().mean()))
#niezblalansowana, 4 z 12 kategorii powyżej średniej, prawodopodobnie nieadaekwatna do populacji; [ptercjalnie dyslkryminująca dla 'housemiad'


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - wstęp cd.</strong></span></h4>
# <ul>
# <li>Zmienna stanu cywilnego nierównoliczna, ale niekoniecznie niereprezentatywna dla populacji  </li>
# <li>Zmienna kredytu o stopie zmiennej jest silnie niezbalansowana - liczne wartości 'unknown', bardzo niskie 'yes'</li>
# <li>Zmienna kredytu hipotecznego dobrze zbalansowana, ale słabo różnicująca</li>
# <li>Zmienna kredytu osobistego słabo zbalansowana</li>
# 
# </ul>

# In[58]:


#jaki rozkład ma zmienna stanu cywilnego
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='marital', data=df, ax=ax,hue="y")
plt.show()
print(df.marital.value_counts())
print('Categories no.: '+ str(df.marital.nunique()))
print('Mean: '+ str(df.marital.value_counts().mean()))
#niezblalansowana, ale niekoniecznie niereprezentatywna dla populacji


# In[59]:


#jaki rozkład ma zmienna posiadania kredytu o zmiennej stopie 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='default', data=df, ax=ax,hue="y")
plt.show()
print(df.default.value_counts())
print('Categories no.: '+ str(df.default.nunique()))
print('Mean: '+ str(df.default.value_counts().mean()))
#niezblalansowana, silna nadreprezentacja osób bez kredytu o zmiennej stopei, duży odstek brak danych, potencjalnie silnie dyskryminująca - tylko 3 odopowiedzi yes    


# In[60]:


#jaki rozkład ma zmienna posiadania kredytu hipotecznego pod nieruchomość
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='housing', data=df, ax=ax,hue="y")
plt.show()
print(df.housing.value_counts())
print('Categories no.: '+ str(df.housing.nunique()))
print('Mean: '+ str(df.housing.value_counts().mean()))
#zblalansowana,ale istotnie różnicująca



# In[61]:


#jaki rozkład ma zmienna posiadania kredytu osobistego
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='loan', data=df, ax=ax,hue="y")
plt.show()
print(df.loan.value_counts())
print('Categories no.: '+ str(df.loan.nunique()))
print('Mean: '+ str(df.loan.value_counts().mean()))
#niezblalasnowana, sałobkorelująca - wykluczona do segmentacji klientów


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - wstęp cd.</strong></span></h4>
# <ul>
# <li>Zmienna wykształcenia silnie zróżnicowana, bardzo mało liczna kategoria 'illeterate' - rekodowano do 'unknown'  </li>
# <li>Zmienna sukcesu zeszłej kampanii marketingowej silnie nierównoliczna - obecna kampania jest kierowana do nowych osób</li>
# <li>Zeszła kampania marketingowa była ponad dwukrotnie skuteczniejsza od , ale obejmowała blisko dziesięciokrotonie liczbę klientów </li>
# <li>Zmienna sposobu kontaktu względnie dobrze zabalasowana i potencjalnie dobrze różnicująca zmienną zależną</li>
# 
# </ul>

# In[62]:


#jaki rozkład ma zmienna wykształcenia
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='education', data=df, ax=ax,hue="y")
plt.show()
print(df.education.value_counts())
print('Categories no.: '+ str(df.education.nunique()))
print('Mean: '+ str(df.education.value_counts().mean()))
#niezblalansowana, silna nadreprezentacja osób z wykształceniem wyższym i licealnym, ptencjalnie silnie dyskryminująca zmienna illetarete,m ale niekoniecznie niereprezentatywna dla populacji


# In[63]:


#jaki rozkład ma zmienna wykształcenia
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='poutcome', data=df, ax=ax,hue="y")
plt.show()
print(df.poutcome.value_counts())
print('Categories no.: '+ str(df.poutcome.nunique()))
print('Mean: '+ str(df.poutcome.value_counts().mean()))
#niezblalansowana, w więszkości klientci są kontaktowani po raz pierszy


# In[64]:


#wynik referencyjny dla porpzedzniej kampani
former_campaign_result=df.poutcome.value_counts()[2]/(df.poutcome.value_counts()[2]+df.poutcome.value_counts()[1])
print('Previous campaign success rate: '+ str(former_campaign_result), 'Current campaign success rate: '+ str(current_campaign_result))
#poprzednia kampania była skuteczniejsza niż obecna


# In[65]:


#jaki rozkład ma zmienna kontaktu
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(10, 3))
sns.countplot(x='contact', data=df, ax=ax,hue="y")
plt.show()
print(df.contact.value_counts())
print('Categories no.: '+ str(df.contact.nunique()))
print('Mean: '+ str(df.contact.value_counts().mean()))
#słabo niezblalansowana, ale potencjalnie dyskryminuąca dobrze klientów



# In[66]:


# do segementacji klientów wykluczono zmienną defulat oraz loan i uznano katorgię illetareate w zbiorze danych education jako błąd  
df.Missing_education.value_counts()


# In[67]:


unknown_values=['unknown','illiterate']
df['Missing_education'] = df['education'].isin(unknown_values)
df.Missing_education.value_counts()


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - segmentacja klientów <strong></span></h4>
# <ul>
# <li>Segemetnację dokonano jedynie w opraciu o zmienne kategorialne i ilościowe bez wymiaru czasowego  </li>
# <li>Opiera się ona na jawnym systemie regułowym w opraciu o średnie, odchylenia standardowe oraz wyrażenia logiczne </li>
# <li>Celem jest określenie grupy najlepszych klientów, do której ma być kierowana nowa kampania oraz wyróżnienia najsłabszych, z których rezygnacja obejmowała by możliwie najmniejszą potencjalną stratę liczby lokat</li>
# <li>Założono, że nowa kampania musi być skierowana do węższej grupie docelowej niż obecna i większej niż poprzdzenia o potencjalnie wyższej skuteczności i łatwości przeprowadzenia</li>
# 
# </ul>

# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - segmentacja klientów cd. <strong></span></h4>
# <ul>
# <li>Do segmentacji opracowana 3 modele w opraciu o % brakujących danych   </li>
#     <li>W ok. 1% braków do dyspozycji mamy wyłącznie zdyskretyzowaną zmienną wieku, w pozostałych 10% dodatkowo stan cywilny oraz zawód, a w pozostałych dodatkowe zmienne</li>
#     <li>Zmienne sortujące w modelu działającym na ok. 90% danych wybrano na podstawie ekploracyjnej analizy danych i ich braków w pozostałej części tj.'job','marital','contact','housing','education','age_bin' </li>
#      <li>'job','marital','contact','education' - silnie dyskryminuje zmienną zależną i wysoki współczynnik korelacji,'age_bin' - dobrze zbalansowana i koresponduje z z pozostałymi modelami: pozostałe zmienne kategorialne nie miały tych cech </li>
#     <li>W wersjach modelu dla najsłabszych klientów wykorzystano zmiene ilościowe 'previous' określającą ilość dni od poprzedniego kontaktu oraz kategorialną 'campaign' określającą czy klient był kontaktowany zgodnie z powyższymi kryteriami </li>
# 
# </ul>

# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - segmentacja klientów cd. <strong></span></h4>
# <ul>
# <li>Dla każdej kombinacji zmiennych sortujących zastosowano współczynniki określające odestek sukcesów dla każdej kombinacji</li>
# <li>Dodatkowo dla każdej kombinacji wyznaczono zmienne określające odestek sukcesów i liczby kontaktów w stosunku do całego zbioru modelu  </li>
# <li>Następnie ustatlono rankingi dla obu powyższych będący ich ilorazem z ich średnimi w całym zbiorze modelu</li>
# <li>Ogólny ranking był ilorazem iloczynów powyższych rankingów z ich średnią w całym zbiorze modelu   </li>
# <li>Miało to na celu ustalenia miary oceny balansującej liczbę sukcesów i liczbę kontaktów w zbiorze, aby nie wybierać zarówno zbyt małych grup o wysokich sukcesach i odwrotnie</li>
# <li>Dodatkowo dla modeli dla najsłabszych klientów ustalono sumaryczną liczbę prób z klientem na podstawie 'previous' i 'campaign'</li>
# </ul>

# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - segmentacja klientów cd. <strong></span></h4>
# <ul>
# <li>Kryterium selekcji klientów najlepszych składało się ze spełnienia 3 warunków logicznych </li>
# <li>3 warunki dla najlepszych klientów : ranking ogólny większy niż średni, odestek sukcesów jest większy niż odestek kontaktów i odstek kontaktów jest większy niż odchylenie standardowe w zbiorze </li>
# <li>Kryterium selekcji klientów najsłabszych składało się ze spełnienia 3 warunków logicznych</li>
# <li>3 warunki dla najsłabszych klientów : ranking ogólny niższy niż średni, odsetek kontaktów jest większy niż odchylenie standardowe po czym również suma prób kontaktów większa niż średnia </li>
# </ul>

# In[ ]:


df_big_segemnts['big_rank_ovr'] < df_big_segemnts['big_rank_ovr'].mean()) &((df_big_segemnts['big_contact_sum_part'] >= df_big_segemnts['big_contact_sum_part'].std())


# In[79]:


# segementacja klientow dla zbioru danych bez braków danych - ranking bazuje na wkaźniku sukcesu i wskaźniku kontatków w stosunku do średnich
df_segemnts=df.query('Missing_job==False  & Missing_marital==False & Missing_housing==False & Missing_education==False & Missing_job==False')
df_segemnts_sum=df_segemnts.groupby(['job','marital','contact','housing','education','age_bin'])['numeric_y'].count().to_frame()
df_segemnts_suc=df_segemnts[df_segemnts['numeric_y'].ne(0)].groupby(['job','marital','contact','housing','education','age_bin'])['numeric_y'].count().to_frame()
df_segments=df_segemnts_suc/df_segemnts_sum
df_segments=df_segments.rename(columns={'numeric_y': 'segment_sucess_rate_ovr'})
df_segments['contact_sum_part']=df_segemnts_sum['numeric_y']/df_segemnts_sum['numeric_y'].sum()
df_segments['success_sum_part']=df_segemnts_suc['numeric_y']/df_segemnts_suc['numeric_y'].sum()
df_segments['contact_ovr_rank']=df_segments['contact_sum_part']/df_segments['contact_sum_part'].mean()
df_segments['success_ovr_rank']=df_segments['segment_sucess_rate_ovr']/df_segments['segment_sucess_rate_ovr'].mean()        
df_segments=df_segments.sort_values(['contact_ovr_rank','success_ovr_rank'],ascending=[False,False])
df_segments=df_segments.dropna()
df_segments['rank_ovr']= (df_segments['contact_ovr_rank']* df_segments['success_ovr_rank'])/(df_segments['contact_ovr_rank']*df_segments['success_ovr_rank']).mean()
df_segments['rank_ovr']=df_segments['rank_ovr']/df_segments['rank_ovr'].mean()
df_segments.sort_values(['rank_ovr','success_ovr_rank'],ascending=[False,False])




# In[80]:


# selekcja najlepszych klientow dla zbioru danych bez braków - kryterium ranking powyżej średniej, liczba kontaktów większa od odchylenia standardowego oraz liczba sukcesów większa od  kontaktów 

segment_matrix_2ord=df_segments[((df_segments['rank_ovr'] > df_segments['rank_ovr'].mean())  & (df_segments['contact_sum_part']>df_segments['contact_sum_part'].std()) & (df_segments['success_sum_part']>df_segments['contact_sum_part']))]
segment_matrix_2ord['premium_client']=True
segment_matrix_2ord


# In[81]:


df_premium=df_segments.merge(segment_matrix_2ord, on=['job','marital','contact','housing','education','age_bin','contact_sum_part','success_sum_part'],  how='outer')
df_premium=df_premium.fillna(False)
df_premium=df_premium.reset_index()
df_premium=df_premium[['job', 'marital', 'contact', 'housing', 'education', 'age_bin', 'premium_client','contact_sum_part','success_sum_part']]


# In[82]:


# segementacja klientow dla zbioru danych z brakami danych dla <1% - ranking bazuje na wkaźniku sukcesu i wskaźniku kontatków w stosunku do średnich - trzy zmienne dzieląca; przedziały wiekowe, zawód i stan cywilny
df_big_segemnts=df.query('Missing_job==False  & Missing_marital==False')
df_big_segemnts_sum=df_big_segemnts.groupby(['job','marital','age_bin'])['numeric_y'].count().to_frame()
df_big_segemnts_suc=df_big_segemnts[df_big_segemnts['numeric_y'].ne(0)].groupby(['job','marital','age_bin'])['numeric_y'].count().to_frame()
df_big_segemnts=df_big_segemnts_suc/df_big_segemnts_sum
df_big_segemnts=df_big_segemnts.rename(columns={'numeric_y': 'big_segment_sucess_rate_ovr'})
df_big_segemnts['big_contact_sum_part']=df_big_segemnts_sum['numeric_y']/df_big_segemnts_sum['numeric_y'].sum()
df_big_segemnts['big_success_sum_part']=df_big_segemnts_suc['numeric_y']/df_big_segemnts_suc['numeric_y'].sum()
df_big_segemnts['big_contact_ovr_rank']=df_big_segemnts['big_contact_sum_part']/df_big_segemnts['big_contact_sum_part'].mean()
df_big_segemnts['big_success_ovr_rank']=df_big_segemnts['big_segment_sucess_rate_ovr']/df_big_segemnts['big_segment_sucess_rate_ovr'].mean()        
df_big_segemnts=df_big_segemnts.sort_values(['big_contact_ovr_rank','big_success_ovr_rank'],ascending=[False,False])
df_big_segemnts=df_big_segemnts.dropna()
df_big_segemnts['big_rank_ovr']= (df_big_segemnts['big_contact_ovr_rank']*df_big_segemnts['big_success_ovr_rank'])/(df_big_segemnts['big_contact_ovr_rank']*df_big_segemnts['big_success_ovr_rank']).mean()
df_big_segemnts['big_rank_ovr']=df_big_segemnts['big_rank_ovr']/df_big_segemnts['big_rank_ovr'].mean()
df_big_segemnts.sort_values(['big_rank_ovr','big_success_ovr_rank'],ascending=[False,False])



# In[83]:


# selekcja najlepszych klientow dla zbioru danych z brakami danych dla <1% - kryterium tożsame jak przy zbiorze bez baku danych
segment_matrix_1ord=df_big_segemnts[( (df_big_segemnts['big_rank_ovr'] > df_big_segemnts['big_rank_ovr'].mean()) & (df_big_segemnts['big_contact_sum_part']> df_big_segemnts['big_contact_sum_part'].std()) & (df_big_segemnts['big_success_sum_part']> df_big_segemnts['big_contact_sum_part'])) ]
segment_matrix_1ord


# In[74]:


#wyznaczenie odestka najlepszych kleintów dla zbioru danych bez braków danych z brakami danych <1%
best_client_1ord=segment_matrix_1ord['big_contact_sum_part'].sum()
best_client_1ord


# In[75]:


#wyznaczenie odestka liczby sukcesów dla wydzielnonych najlepszych klientów zbioru danych z brakami danych <1%
best_success_1ord=segment_matrix_1ord['big_success_sum_part'].sum()
best_success_1ord


# In[76]:


#wyznaczenie odestka najlepszych kleintów dla zbioru danych bez braków danych
best_client_2ord=segment_matrix_2ord['contact_sum_part'].sum()
best_client_2ord


# In[77]:


#wyznaczenie odestka liczby sukcesów dla wydzielnonych najlepszych klientów zbioru danych bez braków danych
best_success_2ord=segment_matrix_2ord['success_sum_part'].sum()
best_success_2ord


# In[78]:


# selekcja najsłabszych klientow dla zbioru danych z brakami <1% - kryterium ranking ponieżej średniej,  wskaźnik kontaktów wyższy niż wkaźnik kontaktów minimalny i segment mnimlaniewyższy od odchylenia standardowego oraz więcej prób kontaktów niż średnia

segment_matrix_2ord_inf=df_segments[(df_segments['rank_ovr'] < df_segments['rank_ovr'].mean())&(( df_segments['contact_sum_part'] >=  df_segments['contact_sum_part'].std()))]
segment_matrix_2ord_inf=segment_matrix_2ord_inf.merge(df[['job','marital','contact','housing','education','age_bin','previous','campaign']],on=['job','marital','contact','housing','education','age_bin'])
segment_matrix_2ord_inf=segment_matrix_2ord_inf.groupby(['job','marital','contact','housing','education','age_bin'])[['rank_ovr','success_sum_part', 'contact_sum_part','previous','campaign']].mean().dropna()
segment_matrix_2ord_inf['sum_of_trails']=segment_matrix_2ord_inf['previous']+segment_matrix_2ord_inf['campaign']
segment_matrix_2ord_inf=segment_matrix_2ord_inf[(segment_matrix_2ord_inf['sum_of_trails']>segment_matrix_2ord_inf['sum_of_trails'].mean())]
segment_matrix_2ord_inf


# In[85]:


# selekcja najsłabszych klientow dla zbioru danych bez braków danych - kryterium ranking ponieżej średniej  i segment dwa razy wyższy od odchylenia standardowego  oraz więcej prób kontaktów niż średnia
segment_matrix_1ord_inf=df_big_segemnts[(df_big_segemnts['big_rank_ovr'] < df_big_segemnts['big_rank_ovr'].mean()) &((df_big_segemnts['big_contact_sum_part'] >= df_big_segemnts['big_contact_sum_part'].std()))]
segment_matrix_1ord_inf=segment_matrix_1ord_inf.merge(df[['job','marital','age_bin','previous','campaign']],on=['job','marital','age_bin'])
segment_matrix_1ord_inf=segment_matrix_1ord_inf.groupby(['job','marital','age_bin'])[['big_rank_ovr','big_success_sum_part', 'big_contact_sum_part','previous','campaign']].mean().dropna()
segment_matrix_1ord_inf['sum_of_trails']=segment_matrix_1ord_inf['previous']+segment_matrix_1ord_inf['campaign']
segment_matrix_1ord_inf=segment_matrix_1ord_inf[(segment_matrix_1ord_inf['sum_of_trails']>segment_matrix_1ord_inf['sum_of_trails'].mean())]
segment_matrix_1ord_inf


# In[86]:


#wyznaczenie odestka nasłabszych kleintów dla zbioru danych z brakami danych <1%
worst_client_1ord=segment_matrix_1ord_inf['big_contact_sum_part'].sum()
worst_client_1ord


# In[87]:


#wyznaczenie odestka liczby sukcesów dla wydzielnonych najlepszych klientów zbioru danych z brakami danych <1%
worst_success_1ord=segment_matrix_1ord_inf['big_success_sum_part'].sum()
worst_success_1ord


# In[88]:


#wyznaczenie odestka nasłabszych kleintów dla zbioru danych bez braków danych
worst_client_2ord=segment_matrix_2ord_inf['contact_sum_part'].sum()
worst_client_2ord


# In[89]:


#wyznaczenie odestka liczby sukcesów dla wydzielnonych najlepszych klientów zbioru danych bez braków danych
worst_success_2ord=segment_matrix_2ord_inf['success_sum_part'].sum()
worst_success_2ord


# In[90]:


# segementacja klientow dla pozostałego 1% zbioru danych  - jedna zmienna dzieląca; przedziały wiekowe
df_small_segments=df.querydf_segments=df.query('Missing_job==True | Missing_marital==True')
df_small_segments_sum=df_small_segments.groupby(['age_bin'])['numeric_y'].count().to_frame()
df_small_segments_suc=df_small_segments[df_small_segments['numeric_y'].ne(0)].groupby(['age_bin'])['numeric_y'].count().to_frame()
df_small_segments=df_small_segments_suc/df_small_segments_sum
df_small_segments=df_small_segments.rename(columns={'numeric_y': 'small_segment_sucess_rate_ovr'})
df_small_segments['small_contact_sum_part']=df_small_segments_sum['numeric_y']/df_small_segments_sum['numeric_y'].sum()
df_small_segments['small_success_sum_part']=df_small_segments_suc['numeric_y']/df_small_segments_suc['numeric_y'].sum()
df_small_segments['small_contact_ovr_rank']=df_small_segments['small_contact_sum_part']/df_small_segments['small_contact_sum_part'].mean()
df_small_segments['small_success_ovr_rank']=df_small_segments['small_segment_sucess_rate_ovr']/df_small_segments['small_segment_sucess_rate_ovr'].mean()        
df_small_segments=df_small_segments.sort_values(['small_contact_ovr_rank','small_success_ovr_rank'],ascending=[False,False])
df_small_segments=df_small_segments.dropna()
df_small_segments['small_rank_ovr']= (df_small_segments['small_contact_ovr_rank']*df_small_segments['small_success_ovr_rank'])/(df_small_segments['small_contact_ovr_rank']*df_small_segments['small_success_ovr_rank']).mean()
df_small_segments['small_rank_ovr']=df_small_segments['small_rank_ovr']/df_small_segments['small_rank_ovr'].mean()
df_small_segments.sort_values(['small_rank_ovr','small_success_ovr_rank'],ascending=[False,False])



# In[91]:


# selekcja najlepszych klientow dla całego zbioru danychh - kryterium tożsame jak w pozostałych przpadakch
segment_matrix_3ord=df_small_segments[((df_small_segments['small_rank_ovr'] > df_small_segments['small_rank_ovr'].mean()) & (df_small_segments['small_contact_sum_part'] > df_small_segments['small_contact_sum_part'].mean()) &  (df_small_segments['small_contact_sum_part'] > df_small_segments['small_contact_sum_part'].mean()) &  (df_small_segments['small_success_sum_part'] > df_small_segments['small_contact_sum_part'])) ]
segment_matrix_3ord.sort_values(['small_rank_ovr'],ascending=[False])


# In[92]:


# selekcja najsłabszych klientow dla całego zbioru danychh - kryterium tożsame jak w pozostałych przpadakach
segment_matrix_3ord_inf=df_small_segments[(df_small_segments['small_rank_ovr'] < df_small_segments['small_rank_ovr'].std()) &  ((df_small_segments['small_contact_sum_part'] > df_small_segments['small_contact_sum_part'].std()))]
segment_matrix_3ord_inf=segment_matrix_3ord_inf.merge(df[['age_bin','previous','campaign']],on=['age_bin'])
segment_matrix_3ord_inf=segment_matrix_3ord_inf.groupby(['age_bin'])[['small_rank_ovr','small_success_sum_part', 'small_contact_sum_part','previous','campaign']].mean().dropna()
segment_matrix_3ord_inf['sum_of_trails']=segment_matrix_3ord_inf['previous']+segment_matrix_3ord_inf['campaign']
segment_matrix_3ord_inf=segment_matrix_3ord_inf[(segment_matrix_3ord_inf['sum_of_trails']>segment_matrix_3ord_inf['sum_of_trails'].mean())]
segment_matrix_3ord_inf.sort_values(['small_rank_ovr'],ascending=[True])


# In[93]:


#wyznaczenie odestka nasłabszych klientów dla pozostałego 1% zbioru danych 
worst_client_3ord=segment_matrix_3ord_inf['small_contact_sum_part'].sum()
worst_client_3ord


# In[94]:


#wyznaczenie odestka sukcesów dla nasłabszych klientów dla pozostałego 1% zbioru danych  
worst_success_3ord=segment_matrix_3ord_inf['small_success_sum_part'].sum()
worst_success_3ord


# In[95]:


#wyznaczenie odestka najelepszych klientów dla pozostałego 1% zbioru danych 
best_client_3ord=segment_matrix_3ord['small_contact_sum_part'].sum()
best_client_3ord


# In[96]:


#wyznaczenie odestka sukcesów dla najlepszych klientów dla pozostałego 1% zbioru danych 
best_success_3ord=segment_matrix_3ord['small_success_sum_part'].sum()
best_success_3ord


# In[97]:


# wyznaczenie odestka brakujących pełnych danych dla całego zbioru
data_without_missing=df.query('Missing_job==False  & Missing_marital==False & Missing_housing==False & Missing_education==False').shape[0]
data_loss=1-(data_without_missing/df.shape[0])
data_loss


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Eksploracja danych (EDA) i inżynieria cech</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Zmienne kategorialne - podsumowanie segmentacja klientów<strong></span></h4>
# <ul>
# <li>Uzyskano wynik dla modelu zbiorczego segmentujący 29.1% najlepszych klientów zakładających 51.1% lokat </li>
# <li>Dla drugiego modelu zbiorczego wskazano 2.1% najsłabszych klientów zakładających 0.2% lokat</li>
# <li>Dokonano ponownej analizy segmentu 29.1% najlepszych klientów celem wydzielenia kryteriow pragmatycznych optymalizujący koszty kolejnej kampanii</li>
# <li>Przy prowadzeniu kampani tylko za pośrednictwem telefonów komórkowych dla grupy klientów posiadających wykształcenie wyższe, zatrudnionych na etacie lub emeryturze i nierozwiedzionych można uzyskać segment 22.1% klientów przynoszących 36.9% lokat</li>
# </ul>

# In[100]:


# sprawdzenia sumaraycznego odestka brakujących danych dla dwóch najrzadszych kolumn <1%
data_without_missing_exept=df.query('Missing_job==True  | Missing_marital==True').shape[0]
data_loss_exept=data_without_missing_exept/df.shape[0]
data_loss_exept


# In[101]:


#odestek klientów najlepszych dla modelu zbiorczego - w zależności od rodzaju braków danych używamy innego modelu
best_client=best_client_1ord*(data_loss-data_loss_exept)+best_client_2ord*(1-data_loss)+best_client_3ord*(data_loss_exept)
(best_client*100).round(1)


# In[102]:


#odestek sukcesów klientów najlepszych dla modelu zbiorczego - w zależności od rodzaju braków danych używamy innego modelu
best_success=best_success_1ord*(data_loss-data_loss_exept)+best_success_2ord*(1-data_loss)+best_success_3ord*(data_loss_exept)
(best_success*100).round(1)


# In[103]:


#odestek klientów najsłabszych do redukcji dla modelu zbiorczego - w zależności od rodzaju braków danych używamy innego modelu
worst_client=worst_client_1ord*(data_loss-data_loss_exept)+worst_client_2ord*(1-data_loss)+worst_client_3ord*(data_loss_exept)
(worst_client*-100).round(1)


# In[104]:


#odestek redukcji sukcesów przy rezygnacji z klientów najsłabszych dla modelu zbiorczego - w zależności od rodzaju braków danych używamy innego modelu
worst_success=worst_success_1ord*(data_loss-data_loss_exept)+worst_success_2ord*(1-data_loss)+worst_success_3ord*(data_loss_exept)
(worst_success*-100).round(1)


# In[105]:


#jaki rozkład ma zmienna wieku u klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='age_bin', data=df_premium, ax=ax, hue="premium_client")
plt.show()


# In[106]:


#jaki rozkład ma zmienna stanu cywilnego u klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='marital', data=df_premium, ax=ax, hue="premium_client")
plt.show()
# w segemntacji uproszocznej odrzucam 'divorced'


# In[107]:


#jaki rozkład ma zmienna zawodu u klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='job', data=df_premium, ax=ax, hue="premium_client")
plt.show()



# In[108]:


# w segemntacji uproszocznej odrzucam 'housemaid', 'self-employed', 'entrepreneur', 'unemployed','student'


# In[109]:


#jaki rozkład ma zmienna typu kontaktu klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='contact', data=df_premium, ax=ax, hue="premium_client")
plt.show()
# w segemntacji uproszocznej odrzucam 'telephone'


# In[110]:


#jaki rozkład ma zmienna kredytu hipotecznego u klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='housing', data=df_premium, ax=ax, hue="premium_client")
plt.show()


# In[111]:


#jaki rozkład ma zmienna wykształcenia u klientów premium 
sns.set(font_scale = 0.8)
fig, ax = plt.subplots(figsize=(12, 3))
sns.countplot(x='education', data=df_premium, ax=ax, hue="premium_client")
plt.show()
# w segemntacji uproszocznej odrzucam wszystkie 'basic'
edu_allowed=['university.degree','high.school','professional.course']


# In[112]:


#kolumny do segmentacji uproszczonej
edu_allowed=['university.degree','high.school','professional.course']
job_not_allowed=['housemaid', 'self-employed', 'entrepreneur', 'unemployed','student']


# In[113]:


#dodatkowe warunki do segmentacji
#założenia:  kampania tylko zapośrednictwem telefonów komórkowych dla grupy klientów posiadających wykształcenie wyższe, zatrudnionych na etacie lub emeryturze, nierozwiedzionych
df_premium_simple=df_premium.loc[(df_premium['contact'] == 'cellular') & (df_premium['marital'] != 'divorced') & (df_premium['education'].isin(edu_allowed))& (~df_premium['job'].isin(job_not_allowed)& (df_premium['premium_client']==True))]



# In[114]:


#wyniki segmentacji
df_premium_simple=df_premium_simple.groupby(['job','marital','contact','housing','education','age_bin'])[['contact_sum_part' , 'success_sum_part']].sum()
df_premium_simple.drop(df_premium_simple[df_premium_simple['contact_sum_part'] == 0].index, inplace = True)
df_premium_simple.sort_values(['contact_sum_part'],ascending=[False])


# In[115]:


#wskaźniki do ewaluacji wyniku przy zastosowaniu uproszczonej segmentacji 
simple_success_premium=df_premium_simple['success_sum_part'].sum()
simple_client_premium=df_premium_simple['contact_sum_part'].sum()


# In[116]:


#cześć klientów premium po segmentacji uproszczonej 
best_client_simple=best_client_1ord*(data_loss-data_loss_exept)+simple_client_premium*(1-data_loss)+best_client_3ord*(data_loss_exept)
(best_client_simple*100).round(1)


# In[117]:


#cześć klientów premium bez segmentacji uproszczonej 
(best_client*100).round(1)


# In[118]:


#cześć sukcesów dla klientów premium po segmentacji uproszczonej 
best_success_simple=best_success_1ord*(data_loss-data_loss_exept)+simple_success_premium*(1-data_loss)+best_success_3ord*(data_loss_exept)
(best_success_simple*100).round(1)


# In[119]:


#cześć sukcesów dla klientów premium bez segmentacji uproszczonej 
(best_success*100).round(1)


# In[120]:


#ilość kategorii dla segemtnacji nieuproszczonej  klientów premium
segment_matrix_2ord.shape[0]


# In[121]:


#ilość kategorii dla segemtnacji uproszczonej  klientów premium
df_premium_simple.shape[0]


# In[122]:


#*****************************************************************************************************************************
#Koniec EDA
#*****************************************************************************************************************************


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele eksplanacyjne - wstęp<strong></span></h4>
# <ul>
# <li>Jako model eksplanacyjny zastosowano algortym XGBClassifier </li>
# <li>Początkowo zastosowano wszystkie zmienne celem wyznaczenia ich ważności do własności objaśniającej modelu </li>
# <li>Zmienne ilościowe wstępnie przeskalowano stosując transformację kwantylową wzglednie odporną na wartości odstające</li>
# <li>Zmienne kategorialne sekwencyjnie zbinaryzowano (One-Hot Encoding), która jest wyczerpująca obliczeniowo, ale umożliwiajwia potencjalną dużą dyskryminację modelu</li>
# <li>Zastosowano logistyczną funkcję celu dla zmiennej zależnej binarnej (binary:logistic), jako głowną metrykę oceny modelu AUC zgodną z dokumentacją bazy danych oraz jako parametr maksymalną głębokości drzewa decyzyjnego przyjęto promil rekordów zbioru </li>
# </ul>

# In[123]:


#*****************************************************************************************************************************
#Start XAI ML- model w pełni eksplanacyjny i selekcja cech
#*****************************************************************************************************************************
#kolumny do tej pory 
df.columns


# In[124]:


#wybieramy kolumny bez informacji pośrednio informujących o zmiennej zależnej jak np. duration, success_rate i nieznanych przed pomiarem jak 'daily_contacts_sum' oraz redundatnych jak age_bin
ml_columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'Missing_job',
       'Missing_marital', 'Missing_education', 'Missing_default',
       'Missing_housing', 'Missing_loan', 'was_contacted', 'first_contacted',
       'once_contacted', 'cumulative_days',
       'year','day', 'euribor3m_ind', 'euribor3m_ind_l',
       'cons_price_ind', 'cons_price_ind_l', 'cons_conf_ind',
       'cons_conf_ind_l', 'emp_var_ind', 'emp_var_ind_l', 'emp_nr_ind',
       'emp_nr_ind_l', 'learning_curve',
       'emp_eff_ind','emp_eff_ind_l','day_of_week_nr','month_number']
ml_df=df[ml_columns]


# In[125]:


#typy zmiennych
ml_df.dtypes


# In[126]:


#podział kolumn po typach zmiennych:
ml_num_columns=['age','campaign', 'pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
               'year','day','cumulative_days','euribor3m_ind','euribor3m_ind_l','cons_price_ind','cons_price_ind_l','cons_conf_ind',
               'cons_price_ind_l','emp_var_ind','emp_var_ind_l','emp_nr_ind','emp_nr_ind_l','learning_curve','emp_eff_ind','day_of_week_nr','month_number']
ml_target=['numeric_y']
ml_cat_columns=['job','marital','education','default','housing','loan','contact','poutcome',
                'Missing_job','Missing_marital','Missing_education','Missing_default','Missing_housing','Missing_loan',
                'was_contacted','first_contacted','once_contacted','month']


# In[127]:


#oznaczenie zmiennej zależnej
y=df[ml_target]


# In[128]:


#zmienne niezależne 
X=ml_df


# In[129]:


#parametry modelu ML
from sklearn.preprocessing import  QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


numerical_transformer = Pipeline(
    steps=[
        ('scaler',  QuantileTransformer()) #standaryzacja odporna na wartości odstające i brak balansu
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')) #kododwanie zmiennych kategorialnych - dokładne, ale obliczeniowo kosztowne
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ml_num_columns),
        ('cat', categorical_transformer, ml_cat_columns)
    ]
)


params = {
    "objective": "binary:logistic",  #zmienna binarna, tylko 1 albo 0 bez prawdopodobieństw
    "eval_metric": "auc",  #metryka referncyjna z artykułu do zbioru danych
    "max_depth": int(df.shape[0]/1000) #maksymalna głębokość drzewa
}

classifier =(params)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', xgb.XGBClassifier())])


# In[130]:


#stworzenie modelu ML  XGB
clf.fit(X, y)


# In[131]:


#stworzenie objaśnienia 
exp = dx.Explainer(clf, X, y)


# In[132]:


#metryka modelu
exp.model_performance(model_type = 'classification')



# In[133]:


#słaby wynik, zgadując zawsze y='no'orzymamy blisko 89% accuracy
1-current_campaign_result


# In[134]:


#metryka ROC 
exp.model_performance(model_type='classification').plot(geom='roc')


# In[135]:


#wyznaczenie wartości cech metodą wyczerpującej permutacji 
feature_importance=exp.model_parts()
feature_importance


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele eksplanacyjne - cechy znaczące<strong></span></h4>
# 

# In[136]:


#Wykres ważności cech
exp.model_parts().plot(max_vars=32)


# In[141]:


#łączenie list cech w grupy funkcjonalne do porównania
import itertools 
temp_list = list(itertools.chain(client_data_columns,campaing_data_columns,context_data_columns,managed_data_columns,missing_data))
feature_eng_columns=list(set(ml_columns) - set(temp_list))
feature_eng_columns


# In[144]:


campaing_data_columns_without_duration=campaing_data_columns
campaing_data_columns_without_duration.remove('duration')


# In[145]:


#model dla cech zgrupowanych - poprawanie wykonana inżynieria cech
exp.model_parts(variable_groups={'client':client_data_columns,'context':context_data_columns,'campaing':campaing_data_columns_without_duration,'managed':managed_data_columns, 'feature_eng':feature_eng_columns, 'missing':missing_data })


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele eksplanacyjne - cechy znaczące zgrupowane<strong></span></h4>
# Zgrupowania zgodnie z dokumentacją, jako 'feature_eng' oraz 'missing' zgrupowano cechy wyprowadzone podczas ekplanacyjnej analizy danych

# In[226]:


#model dla cech zgrupowanych - wykres
exp.model_parts(variable_groups={'client':client_data_columns,'context':context_data_columns,'campaing':campaing_data_columns_without_duration,'managed':managed_data_columns, 'feature_eng':feature_eng_columns, 'missing':missing_data }).plot()


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele eksplanacyjne - podsumowanie<strong></span></h4>
# <ul>
# <li>Model przy użyciu wyczerpującej metody permutacyjnej potwierdził ważność cech wyprowadzonych podczas eksploracyjnej analizy danych </li>
# <li>Na podstawie rankingu ważności cech, wyróżniono cechy słabe,  które w następnej części analizy będą podlegały redukcji wymiarowej</li>
# <li>Dodatkowo sprawdzono model pod względem sprawiedliwości (fairness) oceny względem zmiennych zawodowych - ze względu na brak spełniania warunków dyskryminacji zrekodowano kateorię 'housemaid' na 'unknown'</li>
# <li>Model również nie spełnia kryterium sprawdliwości dla innych zmiennych, ale ze wzgledu na brak wytycznych nie dokonowywano dalszej analizy i modyfikacji modelu w tym aspekcie </li>
# </ul>

# In[147]:


#sprawdzenie sprawiedliwości modelu - czy model preferuje osoby o danym zawodzie, który zwaiera w sobie informacje płci np. housemaid
protected = X.job 
privileged='unknown'
fobject = exp.model_fairness(protected = protected, privileged=privileged)
fobject.fairness_check(epsilon = 0.8)#domyślny referencyjny poziom sprawiedliwości modelu  
#model nie jest sprawiedliwy dla w 3 kryteriach z 5; model STP spełnia tylko admin; FPR żaden, TPR retired, student i unemployed;
# dla przykładu, jeżeli osoba jest sklasyfikowana jako 'unemployed' albo jako 'unknown' model będzie oceniał ją lepiej niż 'housemaid' -  decyzja o zmianie 'housemaid' ze względu na informację o płci do 'unknown'


# In[148]:


# wykres metryk uczciwości 
fig=fobject.plot(show=False)

fig.update_layout(
    autosize=False,
    width=800,
    height=1800,)


fig.show()


# In[149]:


#cech o słabym znaczeniu dla modelu 
weak_features_columns=['Missing_default','month','learning_curve','first_contacted','emp_nr_ind','emp_eff_ind_l','was_contacted','day_of_week','cons_conf_ind_l','year','Missing_housing' ,'Missing_loan','Missing_marital','Missing_job']


# In[150]:


#zamiana wartości zawierającej informacje o płci na nieznaną ze względu na dyskryminaje
df_corrected=df.replace("housemaid", 'unknown')
#ogólna kategoria 'other' dla 'unknown', 
df_corrected=df_corrected.replace('unknown', 'other')


# In[152]:


#*****************************************************************************************************************************
#KONIEC XAI ML - model w pełni eksplanacyjny i selekcja cech
#*****************************************************************************************************************************


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele predykcyjne - wstęp <strong></span></h4>
# <ul>
# <li>Jako modele predykcyjne wykorzystywano algortymy z rodzinych technik oprartych o drzewa decyzyjne XGBClassifier, AdaBoostClassifier oraz RandomForestClassifier</li>
# <li>Dla wszystkich modeli dodano pochodne zmienne otrzymane z techniki redukcji wymiarowej FAMD przeprowadzonej na zmiennych o słabym znaczeniu objaśniającym w porpzdniej części analizy - należy nadmienić, że to powoduje zmniejszenie wartości ekplanacyjnej modeli na rzecz zwiększenia zdolności predykcyjnej  </li>
# <li>Najlepszy model predykcyjny oraz XGBClassifier porównano z modelem ekplnacyjnym opratym również o XGBClassifier </li>
# <li>Najlepszy model oceniono metodą walidacji krzyżowej przy podziale zbioru zgodnie z szeregiem czasowym występującym w zbiorze - ma to na celu uniknięcia błędnej oceny własności predykcyjnej modelu z uwagi na dostęp do informacji z przyszłego elementu szeregu czasowego do przewidywania zmiennej zależnej w aktualnym kroku czasowym  </li>
# </ul>
# <img src="cvts.png" width="1000" height="1000" align='center'>   

# In[162]:


#*****************************************************************************************************************************
#START ML PRED - model predykcyjny  z walidacją krzyżową w domenie czasu 
#*****************************************************************************************************************************


# In[163]:


#redukcja wymiarów zmiennych o słabym znaczeniu dla modelu - FAMD metoda dla zmiennych kategorialnych i ilościowych


# In[164]:


df_famd=df_corrected[weak_features_columns]


# In[165]:


df_famd=df_famd.infer_objects()


# In[166]:


from prince import FAMD
famd = prince.FAMD(n_components=int(len(weak_features_columns)**0.5), n_iter = int(len(weak_features_columns)**0.5)**2, engine="sklearn" )
famd.fit(df_famd)
famd.transform(df_famd)


# In[167]:


famd.eigenvalues_summary


# In[168]:


famd.row_coordinates(df_famd)


# In[169]:


df_corrected=df_corrected[ml_columns]


# In[170]:


df_corrected


# In[171]:


#odrzucenie ze zbioru zmiennych poddanych reukcji wymiarowej 
df_corrected=df_corrected.drop(weak_features_columns,axis=1)


# In[172]:


famd.row_coordinates(df_famd)


# In[173]:


df_corrected=df_corrected.merge(famd.row_coordinates(df_famd),left_index=True, right_index=True)


# In[174]:


df_corrected=df_corrected.rename(columns={0: "FAMD_1", 1: "FAMD_2",2:'FAMD_3' })


# In[175]:


final_columns=['age','job','marital','education','default','housing','loan','contact','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','Missing_education','once_contacted','cumulative_days','day','euribor3m_ind','euribor3m_ind_l','cons_price_ind','cons_price_ind_l','cons_conf_ind','emp_var_ind','emp_var_ind_l','emp_nr_ind_l','emp_eff_ind','day_of_week_nr','month_number','FAMD_1','FAMD_2','FAMD_3']
df_corrected[final_columns]


# In[176]:


df_corrected.dtypes


# In[177]:


final_cat_columns=['job','marital','education','default','housing','loan','contact','poutcome',
                 'Missing_education','once_contacted']
final_num_columns=list(set(final_columns) - set(final_cat_columns))
final_num_columns


# In[178]:


final_num_columns


# In[179]:


#parametry modelu ML
from sklearn.preprocessing import  QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
numerical_transformer = Pipeline(
    steps=[
        ('scaler',  QuantileTransformer()) #standaryzacja odporna na wartości odstające i brak balansu
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')) #kododwanie zmiennych kategorialnych - dokładne, ale obliczeniowo kosztowne
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, final_num_columns),
        ('cat', categorical_transformer, final_cat_columns)
    ]
)
params = {
    "objective": "binary:logistic",  #zmienna binarna, tylko 1 albo 0 bez prawdopodobieństw
    "eval_metric": "auc",  #metryka referncyjna z artykułu do zbioru danych
    "max_depth": int(df.shape[0]/1000) #maksymalna głębokość drzewa
}
classifier =(params)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', xgb.XGBClassifier())])


# In[180]:


#kolejne modele
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
X_fin=df_corrected
XGB_ml = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', xgb.XGBClassifier())])
ADA_ml = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', AdaBoostClassifier())])
RF_ml = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])




# In[181]:


#dopasowania
XGB_ml.fit(X_fin, y)
ADA_ml.fit(X_fin, y)
RF_ml.fit(X_fin, y)



# In[182]:


#dopasowania
XGB_exp = dx.Explainer(XGB_ml, X_fin, y)
ADA_exp = dx.Explainer(ADA_ml, X_fin, y)
RF_exp = dx.Explainer(RF_ml, X_fin, y)


# In[183]:


#wynik XGB
XGB_exp.model_performance(model_type = 'classification')


# In[184]:


#wynik poprzedniego XGB
exp.model_performance(model_type = 'classification')


# In[185]:


#wynik ADA
ADA_exp.model_performance(model_type = 'classification')


# In[186]:


#wynik RF
RF_exp.model_performance(model_type = 'classification')


# In[187]:


XGB_exp.model_parts()


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele predykcyjne - redukcja wymiarowa w kontekście eksplanacyjnym <strong></span></h4>
# Dokonano redukcji wymiarowej z 13 najsłabszych zmiennych do 3 numerycznych metodą FAMD przeznaczoną do użycia dla zmiennych kategorialnych i numerycznych potwiedzając ich zasadność na tle rankingu ważności cech 
# 

# In[188]:


#poprawność redukcji wymiarowej
XGB_exp.model_parts().plot(max_vars=32)


# In[191]:


#przygotowanie danych do modelu predykcyjnego szeregów czasowych
X_fin_date=X_fin.merge(df['date'], left_index=True,right_index=True )


# In[192]:


y_date=y.merge(df['date'], left_index=True,right_index=True )


# In[193]:


X_fin_date.set_index('date', inplace=True) 


# In[194]:


y_date.set_index('date', inplace=True)


# In[195]:


y_date


# In[196]:


#zbiór danych z indexem date 
X_fin_date


# In[197]:


#podział do walidacji krzyżowej szeregów czasowych
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 3)#proporcjonalnie do każdego roku
tss2 = TimeSeriesSplit(n_splits = 26)#proporcjonalnie do każdego miesiąca


# In[198]:


for train_index, test_index in tss.split(X_fin_date):
    X_fin_date_train, X_fin_date_test = X_fin_date.iloc[train_index, :], X_fin_date.iloc[test_index,:]
    y_date_train, y_date_test = y_date.iloc[train_index], y_date.iloc[test_index]
for train_index, test_index in tss2.split(X_fin_date):
    X_fin_date_train, X_fin_date_test = X_fin_date.iloc[train_index, :], X_fin_date.iloc[test_index,:]
    y_date_train, y_date_test = y_date.iloc[train_index], y_date.iloc[test_index]    
    


# In[199]:


#wyniki walidacji krzyżowej dla metryki AUC
from sklearn.model_selection import cross_validate

scores_year = cross_validate(RF_ml, X_fin_date, y_date,
                         scoring='roc_auc', cv=tss,)

scores_month = cross_validate(RF_ml, X_fin_date, y_date,
                         scoring='roc_auc', cv=tss2,)



# In[200]:


#dużo mniejszy wynik według metryki AUC
scores_year['test_score']


# In[201]:


#dużo mniejszy wynik według metryki AUC
scores_month['test_score']


# In[202]:


#wyniki walidacji krzyżowej dla metryki Accuracy
accu_scores_year = cross_validate(RF_ml, X_fin_date, y_date,
                         scoring='accuracy', cv=tss,)

accu_scores_month = cross_validate(RF_ml, X_fin_date, y_date,
                         scoring='accuracy', cv=tss2,)


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele predykcyjne - wynik walidacji krzyżowej w funkcji czasu <strong></span></h4>
# Walidacji dokonano na najlepszym dotychczasowym modelu tj. RandomForestClassifier dla podziałow odpowiadającym liczbie miesięcy i lat  
# 

# In[203]:


#wynik lepszy niż referencyjny w 2 na 3 podziałach - po półtora roku otrzymalibyśmy lepsze przewidywania niż poziom referencyjny  0.89
accu_scores_year['test_score']


# In[204]:


#wynik lepszy niż referencyjny w 20 na 26 podziałach - po ok. pół roku otrzymalibyśmy lepsze przewidywania niż poziom referencyjny 0.89
accu_scores_month['test_score']


# In[205]:


X_fin_date


# In[206]:


#próba zbalansowania zbiorów eksplacyjnego 
from imblearn import over_sampling
sampler = over_sampling.RandomOverSampler()
X_resampled, y_resampled = sampler.fit_resample(X_fin, y)


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele predykcyjne - zbalansowanie zbioru w kontekście eksplanacyjnym <strong></span></h4>
# Dokonano dodatkowo ponownego próbkowania zbioru wyrównującj liczby instancji kategorii w zmiennej zależnej i podwajając wielkość zbiorow danych metodą poprawiając tym samym metryki sukcesu i według nowego poziomu referencyjnego dla 'accuracy'=0.5
# 

# In[207]:


#zbalansowany model eksplanacyjny - poziom referencyjny dla accuracy=0.5
XGB_bal_ml=XGB_ml

XGB_bal_ml.fit(X_resampled, y_resampled)
XGB_bal_ml_exp = dx.Explainer(XGB_bal_ml, X_resampled, y_resampled)
XGB_bal_ml_exp.model_performance(model_type = 'classification')




# In[208]:


#stary po redukcji wymiarowej  - poziom referencyjny dla accuracy=0.89
XGB_exp.model_performance(model_type = 'classification')


# In[209]:


#zbalansowany model ma dużo lepsze parametry ponieważ np. accuracy odnosi się teraz do poziomu referencyjnego 0.5
y_resampled.value_counts()


# In[210]:


XGB_bal_ml_exp.model_parts()


# In[211]:


#nowa ważność zmiennych
XGB_bal_ml_exp.model_parts().plot(max_vars=36)


# In[212]:


#stara ważność zmiennych
XGB_exp.model_parts().plot(max_vars=36)
#zbalansowanie modelu powoduje zwiększenia znaczenia zmiennych jawnych 


# In[213]:


ADA_bal_ml=ADA_ml

ADA_bal_ml.fit(X_resampled, y_resampled)
ADA_bal_ml_exp = dx.Explainer(ADA_bal_ml, X_resampled, y_resampled)
ADA_bal_ml_exp.model_performance(model_type = 'classification')


# In[214]:


RF_bal_ml=RF_ml

RF_bal_ml.fit(X_resampled, y_resampled)
RF_bal_ml_exp = dx.Explainer(RF_bal_ml, X_resampled, y_resampled)
RF_bal_ml_exp.model_performance(model_type = 'classification')


# In[215]:


#najlepszy model eksplanacyjny - zmienne po redukcji wymiarowej zajmują miejsca znacznie niżej po zblansowaniu zbioru przez dodanie próbek
RF_bal_ml_exp.model_parts()


# In[216]:


RF_bal_ml_exp.model_parts().plot(max_vars=36)


# In[217]:


#random forest do balansowania zbioru odpowiedni dla walidacji krzyżowej w szergach czasowaych
from imblearn.ensemble import BalancedRandomForestClassifier
BRF_ml = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', BalancedRandomForestClassifier(sampling_strategy="all", replacement=True))])


# In[218]:


#wyniki walidacji krzyżowej dla metryki AUC
from sklearn.model_selection import cross_validate

bl_scores_year = cross_validate(BRF_ml, X_fin_date, y_date,
                         scoring='roc_auc', cv=tss,)

bl_scores_month = cross_validate(BRF_ml, X_fin_date, y_date,
                         scoring='roc_auc', cv=tss2,)


# In[220]:


bl_scores_year['test_score']


# In[221]:


bl_scores_month['test_score']


# In[222]:


#wyniki walidacji krzyżowej dla metryki Accuracy
bl_accu_scores_year = cross_validate(BRF_ml, X_fin_date, y_date,
                         scoring='accuracy', cv=tss,)

bl_accu_scores_month = cross_validate(BRF_ml, X_fin_date, y_date,
                         scoring='accuracy', cv=tss2,)


# In[223]:


#wyniki walidacji accuracy odnoszą się do poziomu referencyjnego 0.5
bl_accu_scores_year['test_score']


# <p style="text-align:center"></p>
# <span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# <h3><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele uczenia maszynowego</strong></span></h3>
# <h4><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Modele predykcyjne - wynik walidacji krzyżowej w funkcji czasu po zbalansowaniu <strong></span></h4>
# Walidacji po zbalansowaniu zbioru dokonano na algorytmem ekwiwalentnym do najlepszym dotychczasowym modelu RandomForestClassifier, czyli BalancedRandomForestClassifier dla podziałow odpowiadającym liczbie miesięcy i lat.   
# 

# In[224]:


#wyniki walidacji accuracy odnoszą się do poziomu referencyjnego 0.5; po 18 podziale wynik stabilny 
bl_accu_scores_month['test_score']


# In[225]:


#wyniki walidacji accuracy odnoszą się do poziomu referencyjnego 0.88; po 9 podziale wynik stabilny 
accu_scores_month['test_score']
#wyniki standardowego RandomForest jest bardziej stabilny w domenie szeregów czasowych 


# <p style="text-align:center">&nbsp;</p>
# 
# <h2><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Możliwości dalszej analizy:</strong></span></h2>
# 
# <ul><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif">
# 		<li>określenie wytycznych do oceny zbioru pod kątem braku dyskryminacji osób (np. czy wiek jest taką zmienną?)</li>
# 		<li>określenie wytycznych do operacjonalizacji zbioru predykcyjnego (np. horyzontu prognozy, częstości aktualizacji)</li>    
# 		<li>zbalansowanie zbioru przez stosowanie technik próbkowania zarówno pod kątem dyskryminacji osób i predykcji </li>
#         <li>dalsze eksperymetowanie przez stosowanie rożnych techni skalowania, normalizacji, regularyzacji, optymalizacji hiperparametrycznej </li></span>
# 	</ul>
# 
# 
# 
# 
# <p>&nbsp;</p>
# 

# <p style="text-align:center"><span style="font-size:36px"><span style="font-family:Lucida Sans Unicode,Lucida Grande,sans-serif"><strong>Dziękuje za uwagę <br /</strong></span></span></p>
