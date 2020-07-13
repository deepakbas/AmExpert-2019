import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import lightgbm as lgb
import itertools
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import metrics
import re
from collections import Counter

#loading cust_transaction data
trans_df = pd.read_csv("customer_transaction_data_am19.csv")
trans_df.isnull().sum()

#loading item data
item_df = pd.read_csv("item_data_am19.csv")
item_df.isnull().sum()
#merge item with trans_data
trans_df = trans_df.merge(item_df, on=["item_id"], how="left")
del item_df

#date variables
trans_df["date"] = pd.to_datetime(trans_df["date"])

trans_df['t_yr'] = trans_df['date'].dt.year
trans_df['t_mon'] = trans_df['date'].dt.month
trans_df['t_wk'] = trans_df['date'].dt.week
trans_df['t_d'] = trans_df['date'].dt.dayofyear
trans_df['t_dwk'] = trans_df['date'].dt.dayofweek
trans_df['YMon'] = trans_df[['t_yr','t_mon']].dot([100,1])
trans_df['Yday'] = trans_df[['t_yr','t_d']].dot([100,1])

trans_df['other_yes'] = np.where(trans_df['other_discount']<0, 1, 0)
trans_df['coup_yes'] = np.where(trans_df['coupon_discount']<0, 1, 0)

categ_map = {"Grocery":0, "Bakery":1, "Skin & Hair Care":2, "Pharmaceutical":3, "Seafood":4, "Packaged Meat":5, "Dairy, Juices & Snacks":6, "Natural Products":7, "Miscellaneous":8, "Prepared Food":9, "Meat":10, "Vegetables (cut)":11, "Travel":12, "Garden":13, "Flowers & Plants":14, "Salads":15, "Restauarant":16}
trans_df["category"] = trans_df["category"].map(categ_map)

brand_map = {"Established":0, "Local":1}
trans_df["brand_type"] = trans_df["brand_type"].map(brand_map)
pd.options.mode.use_inf_as_na = True

#per quantity feats
trans_df['sp_qty']= trans_df['selling_price']/trans_df['quantity']
trans_df['other_dis_qty']= trans_df['other_discount']/trans_df['quantity']
trans_df['coup_dis_qty']= trans_df['coupon_discount']/trans_df['quantity']
trans_df['tot_dis_qty']= trans_df['other_dis_qty'] + trans_df['coup_dis_qty']
trans_df['pp_qty']= trans_df['sp_qty'] + trans_df['tot_dis_qty']
trans_df['dis_p']= (trans_df['sp_qty'] / trans_df['pp_qty']-1)
trans_df['cdis_p']= trans_df['coupon_discount'] / trans_df['other_discount']

trans_df = trans_df.replace(np.inf, np.nan)

#campwise trans to avoid leakages
trans_df = trans_df.set_index(['date'])

C26=trans_df.loc['2012-8-12':'2012-9-21']
C27=trans_df.loc['2012-8-25':'2012-10-27']
C28=trans_df.loc['2012-9-16':'2012-11-16']
C29=trans_df.loc['2012-10-8':'2012-11-30']
C30=trans_df.loc['2012-11-19':'2013-1-4']
C1=trans_df.loc['2012-12-12':'2013-1-18']
C2=trans_df.loc['2012-12-17':'2013-1-18']
C3=trans_df.loc['2012-12-22':'2013-2-16']
C4=trans_df.loc['2013-1-7':'2013-2-8']
C5=trans_df.loc['2013-1-12':'2013-2-15']
C6=trans_df.loc['2013-1-28':'2013-3-1']
C7=trans_df.loc['2013-2-2':'2013-3-8']
C8=trans_df.loc['2013-2-16':'2013-4-5']
C9=trans_df.loc['2013-3-11':'2013-4-12']
C10=trans_df.loc['2013-4-8':'2013-5-10']
C11=trans_df.loc['2013-4-22':'2013-6-7']
C12=trans_df.loc['2013-4-22':'2013-5-24']
C13=trans_df.loc['2013-5-19':'2013-7-5']
C16=trans_df.loc['2013-7-15':'2013-8-16']
C17=trans_df.loc['2013-7-29':'2013-8-30']
C18=trans_df.loc['2013-8-10':'2013-10-4']
C19=trans_df.loc['2013-8-26':'2013-9-27']
C20=trans_df.loc['2013-9-7':'2013-11-16']
C21=trans_df.loc['2013-9-16':'2013-10-18']
C22=trans_df.loc['2013-9-16':'2013-10-18']
C23=trans_df.loc['2013-10-8':'2013-11-15']
C24=trans_df.loc['2013-10-21':'2013-12-20']
C25=trans_df.loc['2013-10-21':'2013-11-22']

C26["campaign_id"] =26
C27["campaign_id"] =27
C28["campaign_id"] =28
C29["campaign_id"] =29
C30["campaign_id"] =30
C1["campaign_id"] =1
C2["campaign_id"] =2
C3["campaign_id"] =3
C4["campaign_id"] =4
C5["campaign_id"] =5
C6["campaign_id"] =6
C7["campaign_id"] =7
C8["campaign_id"] =8
C9["campaign_id"] =9
C10["campaign_id"] =10
C11["campaign_id"] =11
C12["campaign_id"] =12
C13["campaign_id"] =13
C16["campaign_id"] =16
C17["campaign_id"] =17
C18["campaign_id"] =18
C19["campaign_id"] =19
C20["campaign_id"] =20
C21["campaign_id"] =21
C22["campaign_id"] =22
C23["campaign_id"] =23
C24["campaign_id"] =24
C25["campaign_id"] =25

C26["camp_ord"] =1
C27["camp_ord"] =2
C28["camp_ord"] =3
C29["camp_ord"] =4
C30["camp_ord"] =5
C1["camp_ord"] =6
C2["camp_ord"] =7
C3["camp_ord"] =8
C4["camp_ord"] =9
C5["camp_ord"] =10
C6["camp_ord"] =11
C7["camp_ord"] =12
C8["camp_ord"] =13
C9["camp_ord"] =14
C10["camp_ord"] =15
C11["camp_ord"] =16
C12["camp_ord"] =17
C13["camp_ord"] =18
C16["camp_ord"] =19
C17["camp_ord"] =20
C18["camp_ord"] =21
C19["camp_ord"] =22
C20["camp_ord"] =23
C21["camp_ord"] =24
C22["camp_ord"] =25
C23["camp_ord"] =26
C24["camp_ord"] =27
C25["camp_ord"] =28

trans_df = pd.concat([C26, C27, C28, C29, C30, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25])
del C26, C27, C28, C29, C30, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25
trans_df = trans_df.reset_index()
###
#camp data
camp_df = pd.read_csv("campaign_data_am19.csv")
camp_df["start_date"] = pd.to_datetime(camp_df["start_date"], format="%d/%m/%y")
camp_df["end_date"] = pd.to_datetime(camp_df["end_date"], format="%d/%m/%y")
camp_df.isnull().sum()
#merge camp with all_data
trans_df = trans_df.merge(camp_df, on=["campaign_id"], how="left")

trans_df['s_yr'] = trans_df['start_date'].dt.year
trans_df['s_mon'] = trans_df['start_date'].dt.month
trans_df['s_wk'] = trans_df['start_date'].dt.week
trans_df['s_d'] = trans_df['start_date'].dt.dayofyear
trans_df['s_dwk'] = trans_df['start_date'].dt.dayofweek
trans_df['s_YMon'] = trans_df[['s_yr','s_mon']].dot([100,1])
trans_df['s_Yday'] = trans_df[['s_yr','s_d']].dot([100,1])

trans_df['e_yr'] = trans_df['end_date'].dt.year
trans_df['e_mon'] = trans_df['end_date'].dt.month
trans_df['e_wk'] = trans_df['end_date'].dt.week
trans_df['e_d'] = trans_df['end_date'].dt.dayofyear
trans_df['e_dwk'] = trans_df['end_date'].dt.dayofweek
trans_df['e_YMon'] = trans_df[['e_yr','e_mon']].dot([100,1])
trans_df['e_Yday'] = trans_df[['e_yr','e_d']].dot([100,1])

trans_df['diff_camp_days']= (trans_df['end_date'] - trans_df['start_date']).dt.days

ctype_map = {"X":0, "Y":1}
trans_df["campaign_type"] = trans_df["campaign_type"].map(ctype_map)
trans_df["campaign_type"].value_counts()
trans_df_sam = trans_df.head(1000)

gc.collect()
gc.collect()
##
#custwise diff of trans in each camp
trans_df_pt = trans_df[['customer_id', 'camp_ord']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt.columns = [str(col) + '_cnt_d' for col in trans_df_pt.columns]
trans_df_pt = trans_df_pt.diff(axis=1)
trans_df_pt = trans_df_pt.iloc[:,1:]

trans_df_pt['TTdiff_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TTdiff_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TTdiff_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TTdiff_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#custwise diff of trans in each camp with coup
trans_df_pt = trans_df[['customer_id', 'camp_ord']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord"], aggfunc=len,dropna=True, fill_value=0)

trans_df_pt_1 = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord']]
trans_df_pt_1 = trans_df_pt_1.pivot_table(index=["customer_id"], columns=["camp_ord"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt_2= trans_df_pt_1.div(trans_df_pt, axis=0)
trans_df_pt_2['CT_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_std'] = trans_df_pt_2.std(axis=1)

trans_df_pt_3 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_3 = trans_df_pt_3.reset_index()
trans_df = trans_df.merge(trans_df_pt_3, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt_2.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CTdiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CTdiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CTdiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CTdiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
##
#custwise diff of trans in each camp, t_wk wise
trans_df_pt = trans_df[['customer_id', 'camp_ord', 't_wk']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord","t_wk"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt.columns = [str(col) + '_cnt_d' for col in trans_df_pt.columns]
trans_df_pt = trans_df_pt.diff(axis=1)
trans_df_pt = trans_df_pt.iloc[:,1:]

trans_df_pt['wkTdiff_min'] = trans_df_pt.min(axis=1)
trans_df_pt['wkTdiff_max'] = trans_df_pt.max(axis=1)
trans_df_pt['wkTdiff_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['wkTdiff_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#custwise diff of trans in each camp with coup t_wk
trans_df_pt = trans_df[['customer_id', 'camp_ord', 't_wk']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord","t_wk"], aggfunc=len,dropna=True, fill_value=0)

trans_df_pt_1 = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 't_wk']]
trans_df_pt_1 = trans_df_pt_1.pivot_table(index=["customer_id"], columns=["camp_ord", "t_wk"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt_2= trans_df_pt_1.div(trans_df_pt, axis=0)
trans_df_pt_2['wkCT_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['wkCT_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['wkCT_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['wkCT_std'] = trans_df_pt_2.std(axis=1)

trans_df_pt_3 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_3 = trans_df_pt_3.reset_index()
trans_df = trans_df.merge(trans_df_pt_3, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt_2.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['wkCTdiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['wkCTdiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['wkCTdiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['wkCTdiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")

##
#custwise diff of trans in each camp, t_dwk wise
trans_df_pt = trans_df[['customer_id', 'camp_ord', 't_dwk']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord","t_dwk"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt.columns = [str(col) + '_cnt_d' for col in trans_df_pt.columns]
trans_df_pt = trans_df_pt.diff(axis=1)
trans_df_pt = trans_df_pt.iloc[:,1:]

trans_df_pt['wkdTdiff_min'] = trans_df_pt.min(axis=1)
trans_df_pt['wkdTdiff_max'] = trans_df_pt.max(axis=1)
trans_df_pt['wkdTdiff_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['wkdTdiff_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#custwise diff of trans in each camp with coup t_dwk
trans_df_pt = trans_df[['customer_id', 'camp_ord', 't_dwk']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord","t_dwk"], aggfunc=len,dropna=True, fill_value=0)

trans_df_pt_1 = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 't_dwk']]
trans_df_pt_1 = trans_df_pt_1.pivot_table(index=["customer_id"], columns=["camp_ord", "t_dwk"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt_2= trans_df_pt_1.div(trans_df_pt, axis=0)
trans_df_pt_2['wkdCT_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['wkdCT_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['wkdCT_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['wkdCT_std'] = trans_df_pt_2.std(axis=1)

trans_df_pt_3 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_3 = trans_df_pt_3.reset_index()
trans_df = trans_df.merge(trans_df_pt_3, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt_2.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['wkdCTdiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['wkdCTdiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['wkdCTdiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['wkdCTdiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")

del trans_df_pt, trans_df_pt_1, trans_df_pt_2, trans_df_pt_3
##
#unique item per transaction
trans_df_pt_1 = pd.pivot_table(trans_df, values='item_id', index='customer_id',columns='camp_ord', aggfunc= 'nunique')
trans_df_pt = trans_df[['customer_id', 'camp_ord']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt_2= trans_df_pt_1.div(trans_df_pt, axis=0)

trans_df_pt_2['TT_uniq_item_p_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_uniq_item_p_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_uniq_item_p_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_uniq_item_p_std'] = trans_df_pt_2.std(axis=1)

trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")

#unique brand per transaction
trans_df_pt_1 = pd.pivot_table(trans_df, values='brand', index='customer_id',columns='camp_ord', aggfunc= 'nunique')
trans_df_pt = trans_df[['customer_id', 'camp_ord']]
trans_df_pt = trans_df_pt.pivot_table(index=["customer_id"], columns=["camp_ord"], aggfunc=len,dropna=True, fill_value=0)
trans_df_pt_2= trans_df_pt_1.div(trans_df_pt, axis=0)

trans_df_pt_2['TT_uniq_brand_p_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_uniq_brand_p_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_uniq_brand_p_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_uniq_brand_p_std'] = trans_df_pt_2.std(axis=1)

trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
del trans_df_pt, trans_df_pt_1, trans_df_pt_2

##

pd.options.mode.use_inf_as_na = True
usr_data = trans_df.groupby("customer_id").agg({"coup_yes": ["sum", "count"], "other_yes": "sum", "item_id": "nunique",
                                                 "brand":"nunique", "category":"nunique",
                                                "YMon": ["mean", "std"], "t_dwk":["mean", "std"],
                                                "t_d": ["min", "max", "mean", "std"],
                                                "quantity": ["min", "max", "mean", "std"],
                                                "sp_qty":["mean", "std"],"other_dis_qty":["mean", "std"],
                                                "coup_dis_qty":["mean", "std"],"tot_dis_qty":["mean", "std"],
                                                "pp_qty":["mean", "std"],"dis_p":["mean", "std"],
                                                "cdis_p":["mean", "std"]})
usr_data.columns = ["total_coup_used", "total_trans", "total_other_coup_used", "unique_item", "unique_brand",
                    "unique_category", "YMon_mean", "YMon_std",
                       "t_dwk_mean", "t_dwk_std", "t_d_min", "t_d_max", "t_d_mean", "t_d_std",
                       "qty_min", "qty_max", "qty_mean", "qty_std",
                       "sp_qty_mean", "sp_qty_std","other_dis_qty_mean", "other_dis_qty_std",
                       "coup_dis_qty_mean", "coup_dis_qty_std","tot_dis_qty_mean", "tot_dis_qty_std",
                       "pp_qty_mean", "pp_qty_std","dis_p_mean", "dis_p_std","cdis_p_mean", "cdis_p_std"]
usr_data = usr_data.reset_index(drop=False)

#unique%  of item, brand, categ
usr_data['cust_item_uniq_p']= usr_data['unique_item']/usr_data['total_trans']
usr_data['cust_brand_uniq_p']= usr_data['unique_brand']/usr_data['total_trans']
usr_data['cust_categ_uniq_p']= usr_data['unique_category']/usr_data['total_trans']

#coupon_ratios
usr_data['cust_coup_used_p']= usr_data['total_coup_used']/usr_data['total_trans']
usr_data['cust_other_coup_used_p']= usr_data['total_other_coup_used']/usr_data['total_trans']
usr_data['cust_coupvsother_used_p']= usr_data['total_coup_used']/usr_data['total_other_coup_used']

#coupon_used _feats
trans_df["unq_item_coup_used"] = trans_df.customer_id.map(trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["item_id"].nunique()).fillna(0)
trans_df_sam = trans_df.head(1000)
trans_df["unq_brand_coup_used"] = trans_df.customer_id.map(trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["brand"].nunique()).fillna(0)
trans_df["unq_categ_coup_used"] = trans_df.customer_id.map(trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["category"].nunique()).fillna(0)

trans_df["last_coup_days"]  = trans_df.customer_id.map(trans_df.loc[(trans_df["t_yr"] == 2013) & (trans_df["coup_yes"] == 1)].groupby("customer_id")["t_d"].apply(lambda x: 184 - x.max()))
trans_df["last_trans_day"]  = trans_df.customer_id.map(trans_df.loc[trans_df["t_yr"] == 2013].groupby("customer_id")["t_d"].apply(lambda x: 184 - x.max()))

##
#customer wise camp id wise, mean std, of sp_qty, other_dis_qty, coup_dis_qty, tot_dis_qty, pp_qty, dis_p, cdis_p
#Tot sp_qty mean
trans_df_pt = pd.pivot_table(trans_df, values='sp_qty', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_sp_qty_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_sp_qty_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_sp_qty_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_sp_qty_mean_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['TT_sp_qty_meandiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_sp_qty_meandiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_sp_qty_meandiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_sp_qty_meandiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
#Coup sp_qty mean
trans_df_pt = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 'sp_qty']]
trans_df_pt = pd.pivot_table(trans_df, values='sp_qty', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['CT_sp_qty_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['CT_sp_qty_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['CT_sp_qty_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['CT_sp_qty_mean_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CT_sp_qty_meandiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_sp_qty_meandiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_sp_qty_meandiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_sp_qty_meandiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
##Tot sp_qty std
trans_df_pt = pd.pivot_table(trans_df, values='sp_qty', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['TT_sp_qty_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_sp_qty_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_sp_qty_std_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_sp_qty_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['TT_sp_qty_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_sp_qty_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_sp_qty_stddiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_sp_qty_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")

#Coup sp_qty std
trans_df_pt = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 'sp_qty']]
trans_df_pt = pd.pivot_table(trans_df, values='sp_qty', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['CT_sp_qty_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['CT_sp_qty_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['CT_sp_qty_std_std'] = trans_df_pt.mean(axis=1)
trans_df_pt['CT_sp_qty_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CT_sp_qty_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_sp_qty_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_sp_qty_stddiff_std'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_sp_qty_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
#Tot coup_dis_qty mean
trans_df_pt = pd.pivot_table(trans_df, values='coup_dis_qty', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_dis_qty_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_dis_qty_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_dis_qty_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_dis_qty_mean_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['TT_coup_dis_qty_meandiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_coup_dis_qty_meandiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_coup_dis_qty_meandiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_coup_dis_qty_meandiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##

##
#coup only coup_dis_qty mean
trans_df_pt = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 'coup_dis_qty']]
trans_df_pt = pd.pivot_table(trans_df, values='coup_dis_qty', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['CT_coup_dis_qty_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['CT_coup_dis_qty_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['CT_coup_dis_qty_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['CT_coup_dis_qty_mean_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CT_coup_dis_qty_meandiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_coup_dis_qty_meandiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_coup_dis_qty_meandiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_coup_dis_qty_meandiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
#Tot coup_dis_qty std
trans_df_pt = pd.pivot_table(trans_df, values='coup_dis_qty', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['TT_coup_dis_qty_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_dis_qty_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_dis_qty_std_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_dis_qty_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['TT_coup_dis_qty_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_coup_dis_qty_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_coup_dis_qty_stddiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_coup_dis_qty_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##
#coup only coup_dis_qty std
trans_df_pt = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 'coup_dis_qty']]
trans_df_pt = pd.pivot_table(trans_df, values='coup_dis_qty', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['CT_coup_dis_qty_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['CT_coup_dis_qty_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['CT_coup_dis_qty_std_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['CT_coup_dis_qty_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CT_coup_dis_qty_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_coup_dis_qty_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_coup_dis_qty_stddiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_coup_dis_qty_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")

##
#Tot cdis_p std
trans_df_pt = pd.pivot_table(trans_df, values='cdis_p', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['TT_cdis_p_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_cdis_p_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_cdis_p_std_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_cdis_p_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['TT_cdis_p_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['TT_cdis_p_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['TT_cdis_p_stddiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['TT_cdis_p_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
##

##
#coup only cdis_p std
trans_df_pt = trans_df.loc[(trans_df["coup_yes"] == 1) ,['customer_id', 'camp_ord', 'cdis_p']]
trans_df_pt = pd.pivot_table(trans_df, values='cdis_p', index='customer_id',columns='camp_ord', aggfunc= np.std)
trans_df_pt['CT_cdis_p_std_min'] = trans_df_pt.min(axis=1)
trans_df_pt['CT_cdis_p_std_max'] = trans_df_pt.max(axis=1)
trans_df_pt['CT_cdis_p_std_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['CT_cdis_p_std_std'] = trans_df_pt.std(axis=1)

trans_df_pt_1 = trans_df_pt.iloc[:,-4:]
trans_df_pt_1 = trans_df_pt_1.reset_index()
trans_df = trans_df.merge(trans_df_pt_1, on=["customer_id"], how="left")

trans_df_pt_2 = trans_df_pt.iloc[:,:-4]
trans_df_pt_2 = trans_df_pt_2.diff(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,1:]

trans_df_pt_2['CT_cdis_p_stddiff_min'] = trans_df_pt_2.min(axis=1)
trans_df_pt_2['CT_cdis_p_stddiff_max'] = trans_df_pt_2.max(axis=1)
trans_df_pt_2['CT_cdis_p_stddiff_mean'] = trans_df_pt_2.mean(axis=1)
trans_df_pt_2['CT_cdis_p_stddiff_std'] = trans_df_pt_2.std(axis=1)
trans_df_pt_2 = trans_df_pt_2.iloc[:,-4:]
trans_df_pt_2 = trans_df_pt_2.reset_index()
trans_df = trans_df.merge(trans_df_pt_2, on=["customer_id"], how="left")
###

#cust_wise last 10 unique transactions days
trans_df['date_ord'] = trans_df['date'].apply(lambda x: x.toordinal())
trans_df = trans_df.sort_values(by=["customer_id", "date_ord"], ascending=False).reset_index(drop=True)
trans_df_sam = trans_df.head(1000)

last_t_df = trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "date_ord"]].drop_duplicates()
last_t_10_df = last_t_df.groupby('customer_id').head(10)
last_t_10_df['t_last'] = last_t_10_df.groupby('customer_id').cumcount()+1
last_t_10_df['t_last'] = 'T_'+last_t_10_df['t_last'].astype(str)
last_t_10_df= last_t_10_df.pivot(index='customer_id', columns='t_last', values='date_ord')
#last_t_10_df= last_t_10_df.pivot(index='customer_id', columns='t_last', values='date_ord').reset_index()
last_t_10_df = last_t_10_df.reindex_axis(['T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10'], axis=1)
last_t_10_df= last_t_10_df.diff(axis=1)
last_t_10_df= last_t_10_df.reset_index()
last_t_10_df = last_t_10_df.drop(['T_1'],axis=1)
del last_t_df
last_t_10_df['TDiff_min'] = last_t_10_df.loc[:, 'T_2':'T_10'].min(axis=1)
last_t_10_df['TDiff_max'] = last_t_10_df.loc[:, 'T_2':'T_10'].max(axis=1)
last_t_10_df['TDiff_mean'] = last_t_10_df.loc[:, 'T_2':'T_10'].mean(axis=1)
last_t_10_df['TDiff_std'] = last_t_10_df.loc[:, 'T_2':'T_10'].std(axis=1)
last_t_10_df = last_t_10_df.drop(last_t_10_df.columns.to_series()["T_2":"T_10"], axis=1)
trans_df = trans_df.merge(last_t_10_df, on=["customer_id"], how="left")
del last_t_10_df
##
#cust_wise last 10 items
last_i_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "item_id"]]
last_i_10_df = last_i_df.groupby('customer_id').head(10)
last_i_10_df['i_last'] = last_i_10_df.groupby('customer_id').cumcount()+1
last_i_10_df['i_last'] = 'I_'+last_i_10_df['i_last'].astype(str)
last_i_10_df= last_i_10_df.pivot(index='customer_id', columns='i_last', values='item_id').reset_index()
last_i_10_df = last_i_10_df.reindex_axis(['customer_id','I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6', 'I_7', 'I_8', 'I_9', 'I_10'], axis=1)
del last_i_df
last_i_10_df['I_uniq'] = last_i_10_df.loc[:, 'I_1':'I_10'].nunique(axis=1)
last_i_10_df = last_i_10_df.drop(last_i_10_df.columns.to_series()["I_1":"I_10"], axis=1)
trans_df = trans_df.merge(last_i_10_df, on=["customer_id"], how="left")
del last_i_10_df

#cust_wise last 10 brand
last_b_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "brand"]]
last_b_10_df = last_b_df.groupby('customer_id').head(10)
last_b_10_df['b_last'] = last_b_10_df.groupby('customer_id').cumcount()+1
last_b_10_df['b_last'] = 'B_'+last_b_10_df['b_last'].astype(str)
last_b_10_df= last_b_10_df.pivot(index='customer_id', columns='b_last', values='brand').reset_index()
last_b_10_df = last_b_10_df.reindex_axis(['customer_id','B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10'], axis=1)
del last_b_df
last_b_10_df['B_uniq'] = last_b_10_df.loc[:, 'B_1':'B_10'].nunique(axis=1)
last_b_10_df = last_b_10_df.drop(last_b_10_df.columns.to_series()["B_1":"B_10"], axis=1)
trans_df = trans_df.merge(last_b_10_df, on=["customer_id"], how="left")
del last_b_10_df

#cust_wise last 10 category
last_c_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "category"]]
last_c_10_df = last_c_df.groupby('customer_id').head(10)
last_c_10_df['c_last'] = last_c_10_df.groupby('customer_id').cumcount()+1
last_c_10_df['c_last'] = 'C_'+last_c_10_df['c_last'].astype(str)
last_c_10_df= last_c_10_df.pivot(index='customer_id', columns='c_last', values='category').reset_index()
last_c_10_df = last_c_10_df.reindex_axis(['customer_id','C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10'], axis=1)
del last_c_df
last_c_10_df['C_uniq'] = last_c_10_df.loc[:, 'C_1':'C_10'].nunique(axis=1)
last_c_10_df = last_c_10_df.drop(last_c_10_df.columns.to_series()["C_1":"C_10"], axis=1)
trans_df = trans_df.merge(last_c_10_df, on=["customer_id"], how="left")
del last_c_10_df

#cust_wise last 10 qty transacted
last_q_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "quantity"]]
last_q_10_df = last_q_df.groupby('customer_id').head(10)
last_q_10_df['q_last'] = last_q_10_df.groupby('customer_id').cumcount()+1
last_q_10_df['q_last'] = 'Q_'+last_q_10_df['q_last'].astype(str)
last_q_10_df= last_q_10_df.pivot(index='customer_id', columns='q_last', values='quantity')
#last_q_10_df= last_q_10_df.pivot(index='customer_id', columns='q_last', values='date_ord').reseq_index()
last_q_10_df = last_q_10_df.reindex_axis(['Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8', 'Q_9', 'Q_10'], axis=1)
last_q_10_df= last_q_10_df.diff(axis=1)
last_q_10_df= last_q_10_df.reset_index()
last_q_10_df = last_q_10_df.drop(['Q_1'],axis=1)
del last_q_df
last_q_10_df['QDiff_min'] = last_q_10_df.loc[:, 'Q_2':'Q_10'].min(axis=1)
last_q_10_df['QDiff_max'] = last_q_10_df.loc[:, 'Q_2':'Q_10'].max(axis=1)
last_q_10_df['QDiff_mean'] = last_q_10_df.loc[:, 'Q_2':'Q_10'].mean(axis=1)
last_q_10_df['QDiff_std'] = last_q_10_df.loc[:, 'Q_2':'Q_10'].std(axis=1)
last_q_10_df = last_q_10_df.drop(last_q_10_df.columns.to_series()["Q_2":"Q_10"], axis=1)
trans_df = trans_df.merge(last_q_10_df, on=["customer_id"], how="left")
del last_q_10_df

trans_df_sam = trans_df.head(1000)

#cust_wise last 10 sp_qty, avg, std
last_sp_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "sp_qty"]]
last_sp_10_des_df = last_sp_df.groupby('customer_id').head(10)
last_sp_10_des_df['sp_last'] = last_sp_10_des_df.groupby('customer_id').cumcount()+1
last_sp_10_des_df['sp_last'] = 'SP_'+last_sp_10_des_df['sp_last'].astype(str)
last_sp_10_des_df= last_sp_10_des_df.pivot(index='customer_id', columns='sp_last', values='sp_qty')
last_sp_10_des_df = last_sp_10_des_df.reindex_axis(['SP_1', 'SP_2', 'SP_3', 'SP_4', 'SP_5', 'SP_6', 'SP_7', 'SP_8', 'SP_9', 'SP_10'], axis=1)

last_sp_10_df= last_sp_10_des_df.diff(axis=1)
last_sp_10_df = last_sp_10_df.drop(['SP_1'],axis=1)

last_sp_10_df['SPDiff_min'] = last_sp_10_df.loc[:, 'SP_2':'SP_10'].min(axis=1)
last_sp_10_df['SPDiff_max'] = last_sp_10_df.loc[:, 'SP_2':'SP_10'].max(axis=1)
last_sp_10_df['SPDiff_mean'] = last_sp_10_df.loc[:, 'SP_2':'SP_10'].mean(axis=1)
last_sp_10_df['SPDiff_std'] = last_sp_10_df.loc[:, 'SP_2':'SP_10'].std(axis=1)

last_sp_10_df['SP_min'] = last_sp_10_des_df.loc[:, 'SP_1':'SP_10'].min(axis=1)
last_sp_10_df['SP_max'] = last_sp_10_des_df.loc[:, 'SP_1':'SP_10'].max(axis=1)
last_sp_10_df['SP_mean'] = last_sp_10_des_df.loc[:, 'SP_1':'SP_10'].mean(axis=1)
last_sp_10_df['SP_std'] = last_sp_10_des_df.loc[:, 'SP_1':'SP_10'].std(axis=1)
last_sp_10_df = last_sp_10_df.drop(last_sp_10_df.columns.to_series()["SP_2":"SP_10"], axis=1)
last_sp_10_df= last_sp_10_df.reset_index()
trans_df = trans_df.merge(last_sp_10_df, on=["customer_id"], how="left")
del last_sp_df, last_sp_10_df,last_sp_10_des_df

#cust_wise last 10 other_dis_qty, avg, std

last_odis_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "other_dis_qty"]]
last_odis_10_des_df = last_odis_df.groupby('customer_id').head(10)
last_odis_10_des_df['odis_last'] = last_odis_10_des_df.groupby('customer_id').cumcount()+1
last_odis_10_des_df['odis_last'] = 'ODIS_'+last_odis_10_des_df['odis_last'].astype(str)
last_odis_10_des_df= last_odis_10_des_df.pivot(index='customer_id', columns='odis_last', values='other_dis_qty')
last_odis_10_des_df = last_odis_10_des_df.reindex_axis(['ODIS_1', 'ODIS_2', 'ODIS_3', 'ODIS_4', 'ODIS_5', 'ODIS_6', 'ODIS_7', 'ODIS_8', 'ODIS_9', 'ODIS_10'], axis=1)

last_odis_10_df= last_odis_10_des_df.diff(axis=1)
last_odis_10_df = last_odis_10_df.drop(['ODIS_1'],axis=1)

last_odis_10_df['ODISDiff_min'] = last_odis_10_df.loc[:, 'ODIS_2':'ODIS_10'].min(axis=1)
last_odis_10_df['ODISDiff_max'] = last_odis_10_df.loc[:, 'ODIS_2':'ODIS_10'].max(axis=1)
last_odis_10_df['ODISDiff_mean'] = last_odis_10_df.loc[:, 'ODIS_2':'ODIS_10'].mean(axis=1)
last_odis_10_df['ODISDiff_std'] = last_odis_10_df.loc[:, 'ODIS_2':'ODIS_10'].std(axis=1)

last_odis_10_df['ODIS_min'] = last_odis_10_des_df.loc[:, 'ODIS_1':'ODIS_10'].min(axis=1)
last_odis_10_df['ODIS_max'] = last_odis_10_des_df.loc[:, 'ODIS_1':'ODIS_10'].max(axis=1)
last_odis_10_df['ODIS_mean'] = last_odis_10_des_df.loc[:, 'ODIS_1':'ODIS_10'].mean(axis=1)
last_odis_10_df['ODIS_std'] = last_odis_10_des_df.loc[:, 'ODIS_1':'ODIS_10'].std(axis=1)
last_odis_10_df = last_odis_10_df.drop(last_odis_10_df.columns.to_series()["ODIS_2":"ODIS_10"], axis=1)
last_odis_10_df= last_odis_10_df.reset_index()
trans_df = trans_df.merge(last_odis_10_df, on=["customer_id"], how="left")
del last_odis_df,last_odis_10_df, last_odis_10_des_df
del trans_df_pt, trans_df_pt_1, trans_df_pt_2
#cust_wise last 10 coup_dis_qty, avg, std
last_cdis_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "coup_dis_qty"]]
last_cdis_10_des_df = last_cdis_df.groupby('customer_id').head(10)
last_cdis_10_des_df['cdis_last'] = last_cdis_10_des_df.groupby('customer_id').cumcount()+1
last_cdis_10_des_df['cdis_last'] = 'CDIS_'+last_cdis_10_des_df['cdis_last'].astype(str)
last_cdis_10_des_df= last_cdis_10_des_df.pivot(index='customer_id', columns='cdis_last', values='coup_dis_qty')
last_cdis_10_des_df = last_cdis_10_des_df.reindex_axis(['CDIS_1', 'CDIS_2', 'CDIS_3', 'CDIS_4', 'CDIS_5', 'CDIS_6', 'CDIS_7', 'CDIS_8', 'CDIS_9', 'CDIS_10'], axis=1)

last_cdis_10_df= last_cdis_10_des_df.diff(axis=1)
last_cdis_10_df = last_cdis_10_df.drop(['CDIS_1'],axis=1)

last_cdis_10_df['CDISDiff_min'] = last_cdis_10_df.loc[:, 'CDIS_2':'CDIS_10'].min(axis=1)
last_cdis_10_df['CDISDiff_max'] = last_cdis_10_df.loc[:, 'CDIS_2':'CDIS_10'].max(axis=1)
last_cdis_10_df['CDISDiff_mean'] = last_cdis_10_df.loc[:, 'CDIS_2':'CDIS_10'].mean(axis=1)
last_cdis_10_df['CDISDiff_std'] = last_cdis_10_df.loc[:, 'CDIS_2':'CDIS_10'].std(axis=1)

last_cdis_10_df['CDIS_min'] = last_cdis_10_des_df.loc[:, 'CDIS_1':'CDIS_10'].min(axis=1)
last_cdis_10_df['CDIS_max'] = last_cdis_10_des_df.loc[:, 'CDIS_1':'CDIS_10'].max(axis=1)
last_cdis_10_df['CDIS_mean'] = last_cdis_10_des_df.loc[:, 'CDIS_1':'CDIS_10'].mean(axis=1)
last_cdis_10_df['CDIS_std'] = last_cdis_10_des_df.loc[:, 'CDIS_1':'CDIS_10'].std(axis=1)
last_cdis_10_df = last_cdis_10_df.drop(last_cdis_10_df.columns.to_series()["CDIS_2":"CDIS_10"], axis=1)
last_cdis_10_df= last_cdis_10_df.reset_index()
trans_df = trans_df.merge(last_cdis_10_df, on=["customer_id"], how="left")
del last_cdis_df,last_cdis_10_df, last_cdis_10_des_df

#cust_wise last 10 dis_p, avg, std
last_disp_df= trans_df.loc[(trans_df["coup_yes"] == 1) , ["customer_id", "dis_p"]]
last_disp_10_des_df = last_disp_df.groupby('customer_id').head(10)
last_disp_10_des_df['disp_last'] = last_disp_10_des_df.groupby('customer_id').cumcount()+1
last_disp_10_des_df['disp_last'] = 'DISP_'+last_disp_10_des_df['disp_last'].astype(str)
last_disp_10_des_df= last_disp_10_des_df.pivot(index='customer_id', columns='disp_last', values='dis_p')
last_disp_10_des_df = last_disp_10_des_df.reindex_axis(['DISP_1', 'DISP_2', 'DISP_3', 'DISP_4', 'DISP_5', 'DISP_6', 'DISP_7', 'DISP_8', 'DISP_9', 'DISP_10'], axis=1)

last_disp_10_df= last_disp_10_des_df.diff(axis=1)
last_disp_10_df = last_disp_10_df.drop(['DISP_1'],axis=1)

last_disp_10_df['DISPDiff_min'] = last_disp_10_df.loc[:, 'DISP_2':'DISP_10'].min(axis=1)
last_disp_10_df['DISPDiff_max'] = last_disp_10_df.loc[:, 'DISP_2':'DISP_10'].max(axis=1)
last_disp_10_df['DISPDiff_mean'] = last_disp_10_df.loc[:, 'DISP_2':'DISP_10'].mean(axis=1)
last_disp_10_df['DISPDiff_std'] = last_disp_10_df.loc[:, 'DISP_2':'DISP_10'].std(axis=1)

last_disp_10_df['DISP_min'] = last_disp_10_des_df.loc[:, 'DISP_1':'DISP_10'].min(axis=1)
last_disp_10_df['DISP_max'] = last_disp_10_des_df.loc[:, 'DISP_1':'DISP_10'].max(axis=1)
last_disp_10_df['DISP_mean'] = last_disp_10_des_df.loc[:, 'DISP_1':'DISP_10'].mean(axis=1)
last_disp_10_df['DISP_std'] = last_disp_10_des_df.loc[:, 'DISP_1':'DISP_10'].std(axis=1)
last_disp_10_df = last_disp_10_df.drop(last_disp_10_df.columns.to_series()["DISP_2":"DISP_10"], axis=1)
last_disp_10_df= last_disp_10_df.reset_index()
trans_df = trans_df.merge(last_disp_10_df, on=["customer_id"], how="left")
del last_disp_df,last_disp_10_df, last_disp_10_des_df

##Most Common features
most_common_brand = trans_df.groupby("customer_id")["brand"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_brand = pd.DataFrame(most_common_brand).reset_index()
most_common_brand=most_common_brand.rename(columns = {'brand':'MCbrand'})
trans_df = trans_df.merge(most_common_brand, on=["customer_id"], how="left")

most_common_brand = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["brand"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_brand = pd.DataFrame(most_common_brand).reset_index()
most_common_brand=most_common_brand.rename(columns = {'brand':'MCbranddis'})
trans_df = trans_df.merge(most_common_brand, on=["customer_id"], how="left")

#
most_common_category = trans_df.groupby("customer_id")["category"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_category = pd.DataFrame(most_common_category).reset_index()
most_common_category=most_common_category.rename(columns = {'category':'MCcategory'})
trans_df = trans_df.merge(most_common_category, on=["customer_id"], how="left")

most_common_category = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["category"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_category = pd.DataFrame(most_common_category).reset_index()
most_common_category=most_common_category.rename(columns = {'category':'MCcategorydis'})
trans_df = trans_df.merge(most_common_category, on=["customer_id"], how="left")

##
most_common_item_id = trans_df.groupby("customer_id")["item_id"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_item_id = pd.DataFrame(most_common_item_id).reset_index()
most_common_item_id=most_common_item_id.rename(columns = {'item_id':'MCitem_id'})
trans_df = trans_df.merge(most_common_item_id, on=["customer_id"], how="left")

most_common_item_id = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["item_id"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_item_id = pd.DataFrame(most_common_item_id).reset_index()
most_common_item_id=most_common_item_id.rename(columns = {'item_id':'MCitem_iddis'})
trans_df = trans_df.merge(most_common_item_id, on=["customer_id"], how="left")

#date most common feats
most_common_YMon = trans_df.groupby("customer_id")["YMon"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_YMon = pd.DataFrame(most_common_YMon).reset_index()
most_common_YMon=most_common_YMon.rename(columns = {'YMon':'MCYMon'})
trans_df = trans_df.merge(most_common_YMon, on=["customer_id"], how="left")

most_common_YMon = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["YMon"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_YMon = pd.DataFrame(most_common_YMon).reset_index()
most_common_YMon=most_common_YMon.rename(columns = {'YMon':'MCYMondis'})
trans_df = trans_df.merge(most_common_YMon, on=["customer_id"], how="left")
##
most_common_t_mon = trans_df.groupby("customer_id")["t_mon"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_mon = pd.DataFrame(most_common_t_mon).reset_index()
most_common_t_mon=most_common_t_mon.rename(columns = {'t_mon':'MCt_mon'})
trans_df = trans_df.merge(most_common_t_mon, on=["customer_id"], how="left")

most_common_t_mon = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["t_mon"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_mon = pd.DataFrame(most_common_t_mon).reset_index()
most_common_t_mon=most_common_t_mon.rename(columns = {'t_mon':'MCt_mondis'})
trans_df = trans_df.merge(most_common_t_mon, on=["customer_id"], how="left")
##
most_common_t_wk = trans_df.groupby("customer_id")["t_wk"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_wk = pd.DataFrame(most_common_t_wk).reset_index()
most_common_t_wk=most_common_t_wk.rename(columns = {'t_wk':'MCt_wk'})
trans_df = trans_df.merge(most_common_t_wk, on=["customer_id"], how="left")

most_common_t_wk = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["t_wk"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_wk = pd.DataFrame(most_common_t_wk).reset_index()
most_common_t_wk=most_common_t_wk.rename(columns = {'t_wk':'MCt_wkdis'})
trans_df = trans_df.merge(most_common_t_wk, on=["customer_id"], how="left")
##
most_common_t_dwk = trans_df.groupby("customer_id")["t_dwk"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_dwk = pd.DataFrame(most_common_t_dwk).reset_index()
most_common_t_dwk=most_common_t_dwk.rename(columns = {'t_dwk':'MCt_dwk'})
trans_df = trans_df.merge(most_common_t_dwk, on=["customer_id"], how="left")

most_common_t_dwk = trans_df.loc[trans_df["coup_yes"] == 1].groupby("customer_id")["t_dwk"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_t_dwk = pd.DataFrame(most_common_t_dwk).reset_index()
most_common_t_dwk=most_common_t_dwk.rename(columns = {'t_dwk':'MCt_dwkdis'})
trans_df = trans_df.merge(most_common_t_dwk, on=["customer_id"], how="left")

del most_common_brand, most_common_category, most_common_item_id, most_common_YMon, most_common_t_mon, most_common_t_wk, most_common_t_dwk
trans_df_sam = trans_df.head(1000)
#coup_yes feats
#Tot coup_yes mean campaign_type wise
trans_df_pt = trans_df.loc[(trans_df["campaign_type"] == 0) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_camp0_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_yes_camp0_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_yes_camp0_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_yes_camp0_mean_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["campaign_type"] == 1) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_camp1_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_yes_camp1_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_yes_camp1_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_yes_camp1_mean_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#Tot coup_yes mean brand_type wise
trans_df_pt = trans_df.loc[(trans_df["brand_type"] == 0) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_brand0_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_yes_brand0_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_yes_brand0_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_yes_brand0_mean_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["brand_type"] == 1) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_brand1_mean_min'] = trans_df_pt.min(axis=1)
trans_df_pt['TT_coup_yes_brand1_mean_max'] = trans_df_pt.max(axis=1)
trans_df_pt['TT_coup_yes_brand1_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt['TT_coup_yes_brand1_mean_std'] = trans_df_pt.std(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-4:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#trans_df = trans_df.drop(['TT_coup_yes_cat1_mean_min', 'TT_coup_yes_cat1_mean_max', 'TT_coup_yes_cat1_mean_mean', 'TT_coup_yes_cat1_mean_std', 'TT_coup_yes_cat2_mean_min', 'TT_coup_yes_cat2_mean_max', 'TT_coup_yes_cat2_mean_mean', 'TT_coup_yes_cat2_mean_std', 'TT_coup_yes_cat3_mean_min', 'TT_coup_yes_cat3_mean_max', 'TT_coup_yes_cat3_mean_mean', 'TT_coup_yes_cat3_mean_std', 'TT_coup_yes_cat4_mean_min', 'TT_coup_yes_cat4_mean_max', 'TT_coup_yes_cat4_mean_mean', 'TT_coup_yes_cat4_mean_std', '16_x', '17_y', '18_y', 'TT_coup_yes_cat5_mean_mean', '16_y', 'TT_coup_yes_cat6_mean_mean'],axis=1)
#trans_df = trans_df.iloc[:,:-2]

#Tot coup_yes mean category wise
trans_df_pt = trans_df.loc[(trans_df["category"] == 0) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat0_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 1) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat1_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 2) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat2_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 3) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat3_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 4) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat4_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 5) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat5_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 6) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat6_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 7) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat7_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 8) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat8_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 9) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat9_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 10) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat10_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 11) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat11_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 12) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat12_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 13) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat13_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 14) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat14_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 15) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat15_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

trans_df_pt = trans_df.loc[(trans_df["category"] == 16) ,['customer_id', 'camp_ord', 'coup_yes']]
trans_df_pt = pd.pivot_table(trans_df_pt, values='coup_yes', index='customer_id',columns='camp_ord', aggfunc= np.mean)
trans_df_pt['TT_coup_yes_cat16_mean_mean'] = trans_df_pt.mean(axis=1)
trans_df_pt = trans_df_pt.iloc[:,-1:]
trans_df_pt = trans_df_pt.reset_index()
trans_df = trans_df.merge(trans_df_pt, on=["customer_id"], how="left")

#Merge feats into trans_df

dtype_trans_df = trans_df.dtypes.to_frame('dtypes').reset_index()
trans_df = trans_df.drop(['date', 'item_id', 'quantity', 'selling_price', 'other_discount', 'coupon_discount', 'brand', 'brand_type', 'category', 't_yr', 't_mon', 't_wk', 't_dwk', 'YMon', 'date_ord', 'sp_qty', 'other_dis_qty', 'coup_dis_qty', 'tot_dis_qty', 'pp_qty', 'dis_p', 'cdis_p'],axis=1)
trans_df = trans_df.drop(['other_yes', 'coup_yes'],axis=1)
trans_df = trans_df.drop(['t_d'],axis=1)
trans_df = trans_df.drop(['Yday'],axis=1)
trans_df = trans_df.drop(['campaign_id', 'camp_ord', 'campaign_type', 'start_date', 'end_date', 's_yr', 's_mon', 's_wk', 's_d', 's_dwk', 's_YMon', 's_Yday', 'e_yr', 'e_mon', 'e_wk', 'e_d', 'e_dwk', 'e_YMon', 'e_Yday', 'diff_camp_days'],axis=1)
trans_df = trans_df.drop_duplicates()
dtype_trans_df = trans_df.dtypes.to_frame('dtypes').reset_index()
pd.options.mode.use_inf_as_na = True
trans_df = trans_df.replace(np.inf, np.nan)

#merge user data
trans_df = trans_df.merge(usr_data, on=["customer_id"], how="left")
trans_df_sam = trans_df.head(1000)
del usr_data
dtype_trans_df = trans_df.dtypes.to_frame('dtypes').reset_index()
gc.collect()
gc.collect()

#Loading the data
train_df = pd.read_csv("train_am19.csv")
test_df = pd.read_csv("test_am19.csv")
train_df.dtypes
test_df.dtypes
train_df.isnull().sum()
test_df.isnull().sum()
train_df['redemption_status'].value_counts()

#traintest
ID_y_col = ["id","redemption_status"]
raw_col = [col for col in train_df.columns if col not in ID_y_col]
train_df["train_set"] = 1
test_df["train_set"] = 0
test_df["redemption_status"] = -9

#Concat Train and Test data for creating different features as both Train and Test have similar distribution
all_df = pd.concat([train_df, test_df])
gc.collect()
gc.collect()

#camp data
camp_df = pd.read_csv("campaign_data_am19.csv")
camp_df["start_date"] = pd.to_datetime(camp_df["start_date"], format="%d/%m/%y")
camp_df["end_date"] = pd.to_datetime(camp_df["end_date"], format="%d/%m/%y")
camp_df.isnull().sum()
#merge camp with all_data
all_df = all_df.merge(camp_df, on=["campaign_id"], how="left")

all_df['s_yr'] = all_df['start_date'].dt.year
all_df['s_mon'] = all_df['start_date'].dt.month
all_df['s_wk'] = all_df['start_date'].dt.week
all_df['s_d'] = all_df['start_date'].dt.day
all_df['s_dwk'] = all_df['start_date'].dt.dayofweek

all_df['s_yr'].value_counts()
all_df['s_mon'].value_counts()
all_df['s_wk'].value_counts()
all_df['s_d'].value_counts()
all_df['s_dwk'].value_counts()

all_df['e_yr'] = all_df['end_date'].dt.year
all_df['e_mon'] = all_df['end_date'].dt.month
all_df['e_wk'] = all_df['end_date'].dt.week
all_df['e_d'] = all_df['end_date'].dt.day
all_df['e_dwk'] = all_df['end_date'].dt.dayofweek

all_df['e_yr'].value_counts()
all_df['e_mon'].value_counts()
all_df['e_wk'].value_counts()
all_df['e_d'].value_counts()
all_df['e_dwk'].value_counts()

all_df['diff_camp_days']= (all_df['end_date'] - all_df['start_date']).dt.days

ctype_map = {"X":0, "Y":1}
all_df["campaign_type"] = all_df["campaign_type"].map(ctype_map)
all_df["campaign_type"].value_counts()

gc.collect()
gc.collect()
dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()

#campaign features-unique %
for col in ["coupon_id", "customer_id"]:
    gdf = all_df.groupby("campaign_id")[col].nunique().reset_index()
    gdf.columns = ["campaign_id", "camp_"+col+"nunique"]
    all_df = pd.merge(all_df, gdf, on="campaign_id", how="left")

#all_df['camp_coup_unique_camptype'].value_counts()

gdf = all_df.groupby(["campaign_id", "coupon_id"])["customer_id"].nunique().reset_index()
gdf.columns = ["campaign_id", "coupon_id", "camp_coup_unique_cust"]
all_df = all_df.merge(gdf, on=["campaign_id", "coupon_id"], how="left")

##
gdf = all_df.groupby(["campaign_id", "customer_id"])["coupon_id"].nunique().reset_index()
gdf.columns = ["campaign_id", "customer_id", "camp_cus_unique_coup"]
all_df = all_df.merge(gdf, on=["campaign_id", "customer_id"], how="left")

##
gdf = all_df.groupby("campaign_id")["id"].count().reset_index()
gdf.columns = ["campaign_id", "camp_id_count"]
all_df = all_df.merge(gdf, on=["campaign_id"], how="left")
##

all_df['camp_coup_uniq_p']= all_df['camp_coupon_idnunique']/all_df['camp_id_count']
all_df['camp_cus_uniq_p']= all_df['camp_customer_idnunique']/all_df['camp_id_count']

all_df['camp_coup_cus_uniq_p']= all_df['camp_coup_unique_cust']/all_df['camp_id_count']
all_df['camp_cus_coup_uniq_p']= all_df['camp_cus_unique_coup']/all_df['camp_id_count']
dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()

#prev, next camps
all_df['start_date_ord'] = all_df['start_date'].apply(lambda x: x.toordinal())
all_df['end_date_ord'] = all_df['end_date'].apply(lambda x: x.toordinal())

#Sorting of data
all_df = all_df.sort_values(by=["customer_id", "start_date_ord"]).reset_index(drop=True)

#Selected 4 shifts as 75% of members have max of 4 visits
#Memberid wise what were the Previous 4 checkins and Next 4 checkins

all_df["cus_str_shift1"] = all_df.groupby("customer_id")["start_date_ord"].shift(1)
all_df["cus_str_shift2"] = all_df.groupby("customer_id")["start_date_ord"].shift(2)
all_df["cus_str_shift3"] = all_df.groupby("customer_id")["start_date_ord"].shift(3)
all_df["cus_str_shiftn1"] = all_df.groupby("customer_id")["start_date_ord"].shift(-1)
all_df["cus_str_shiftn2"] = all_df.groupby("customer_id")["start_date_ord"].shift(-2)
all_df["cus_str_shiftn3"] = all_df.groupby("customer_id")["start_date_ord"].shift(-3)

all_df["diff1_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shift1']
all_df["diff2_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shift2']
all_df["diff3_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shift3']

all_df["diffn1_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shiftn1']
all_df["diffn2_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shiftn2']
all_df["diffn3_cus_str"] = all_df['start_date_ord'] - all_df['cus_str_shiftn3']

all_df = all_df.sort_values(by=["customer_id", "campaign_id", "start_date_ord"]).reset_index(drop=True)

all_df["cus_camp_shift1"] = all_df.groupby("customer_id")["campaign_id"].shift(1)
all_df["cus_camp_shift2"] = all_df.groupby("customer_id")["campaign_id"].shift(2)
all_df["cus_camp_shift3"] = all_df.groupby("customer_id")["campaign_id"].shift(3)
all_df["cus_camp_shiftn1"] = all_df.groupby("customer_id")["campaign_id"].shift(-1)
all_df["cus_camp_shiftn2"] = all_df.groupby("customer_id")["campaign_id"].shift(-2)
all_df["cus_camp_shiftn3"] = all_df.groupby("customer_id")["campaign_id"].shift(-3)

all_df = all_df.sort_values(by=["customer_id", "coupon_id", "start_date_ord"]).reset_index(drop=True)

all_df["cus_coup_shift1"] = all_df.groupby("customer_id")["coupon_id"].shift(1)
all_df["cus_coup_shift2"] = all_df.groupby("customer_id")["coupon_id"].shift(2)
all_df["cus_coup_shift3"] = all_df.groupby("customer_id")["coupon_id"].shift(3)
all_df["cus_coup_shiftn1"] = all_df.groupby("customer_id")["coupon_id"].shift(-1)
all_df["cus_coup_shiftn2"] = all_df.groupby("customer_id")["coupon_id"].shift(-2)
all_df["cus_coup_shiftn3"] = all_df.groupby("customer_id")["coupon_id"].shift(-3)

all_df = all_df.sort_values(by=["customer_id", "start_date_ord"]).reset_index(drop=True)

all_df["cus_end_shift1"] = all_df.groupby("customer_id")["end_date_ord"].shift(1)
all_df["cus_end_shift2"] = all_df.groupby("customer_id")["end_date_ord"].shift(2)
all_df["cus_end_shift3"] = all_df.groupby("customer_id")["end_date_ord"].shift(3)
all_df["cus_end_shiftn1"] = all_df.groupby("customer_id")["end_date_ord"].shift(-1)
all_df["cus_end_shiftn2"] = all_df.groupby("customer_id")["end_date_ord"].shift(-2)
all_df["cus_end_shiftn3"] = all_df.groupby("customer_id")["end_date_ord"].shift(-3)

##
all_df["diff1_cus_end"] = all_df['cus_end_shift1'] - all_df['start_date_ord']
all_df["diff2_cus_end"] = all_df['cus_end_shift2'] - all_df['cus_str_shift1']
all_df["diff3_cus_end"] = all_df['cus_end_shift3'] - all_df['cus_str_shift2']

all_df["diffn1_cus_end"] = all_df['cus_end_shiftn1'] - all_df['start_date_ord']
all_df["diffn2_cus_end"] = all_df['cus_end_shiftn2'] - all_df['cus_str_shiftn1']
all_df["diffn3_cus_end"] = all_df['cus_end_shiftn3'] - all_df['cus_str_shiftn2']

dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()

del camp_df, gdf, col, ctype_map
#cust_demo data

#coupon_item data
coup_df = pd.read_csv("coupon_item_mapping_am19.csv")
coup_df.isnull().sum()
for c in coup_df.columns:
    print ("---- %s ---" % c)
    print (coup_df[c].value_counts())

item_df = pd.read_csv("item_data_am19.csv")
item_df.isnull().sum()
for c in item_df.columns:
    print ("---- %s ---" % c)
    print (item_df[c].value_counts())

#merge item with coup data
coup_df = coup_df.merge(item_df, on=["item_id"], how="left")
#coup_item features

gdf = coup_df.groupby("coupon_id")["brand"].nunique().reset_index()
gdf.columns = ["coupon_id", "coup_brand_uniq"]
coup_df = coup_df.merge(gdf, on=["coupon_id"], how="left")

gdf = coup_df.groupby("coupon_id")["brand"].count().reset_index()
gdf.columns = ["coupon_id", "coup_brand_cnt"]
coup_df = coup_df.merge(gdf, on=["coupon_id"], how="left")

coup_df['coup_brand_uniq_p']= coup_df['coup_brand_uniq']/coup_df['coup_brand_cnt']

##
gdf = coup_df.groupby("coupon_id")["category"].nunique().reset_index()
gdf.columns = ["coupon_id", "coup_categ_uniq"]
coup_df = coup_df.merge(gdf, on=["coupon_id"], how="left")

gdf = coup_df.groupby("coupon_id")["category"].count().reset_index()
gdf.columns = ["coupon_id", "coup_categ_cnt"]
coup_df = coup_df.merge(gdf, on=["coupon_id"], how="left")

coup_df['coup_categ_uniq_p']= coup_df['coup_categ_uniq']/coup_df['coup_categ_cnt']

##
most_common_brand = coup_df.groupby("coupon_id")["brand"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_brand = pd.DataFrame(most_common_brand).reset_index()
most_common_brand=most_common_brand.rename(columns = {'brand':'MCCoupbrand'})
coup_df = coup_df.merge(most_common_brand, on=["coupon_id"], how="left")

most_common_categ = coup_df.groupby("coupon_id")["category"].apply(lambda x: Counter(x).most_common(1)[0][0])
most_common_categ = pd.DataFrame(most_common_categ).reset_index()
most_common_categ=most_common_categ.rename(columns = {'category':'MCCoupcategory'})
coup_df = coup_df.merge(most_common_categ, on=["coupon_id"], how="left")

#most_common_categ_dum = pd.get_dummies(most_common_categ, columns=['MCcategory'])
coup_df = pd.get_dummies(coup_df, columns=['brand_type'])

coup_df['category_c'] = coup_df['category']
coup_df = pd.get_dummies(coup_df, columns=['category'])

coup_df['MCCoupcategory_c'] = coup_df['MCCoupcategory']
coup_df = pd.get_dummies(coup_df, columns=['MCCoupcategory'])

coup_df_pt = coup_df.pivot_table(index='coupon_id', columns='category_c', aggfunc='size', fill_value=0)
coup_df_pt.columns = [str(col) + '_cnt' for col in coup_df_pt.columns]
coup_df_pt = coup_df_pt.reset_index()
coup_df = coup_df.merge(coup_df_pt, on=["coupon_id"], how="left")

del coup_df_pt, most_common_brand, most_common_categ, item_df, gdf
dtype_coup = coup_df.dtypes.to_frame('dtypes').reset_index()

coup_df = coup_df.drop(['item_id', 'brand','category_c'],axis=1)
coup_df = coup_df.drop_duplicates()

for col in ["category_Bakery", "category_Dairy, Juices & Snacks", "category_Flowers & Plants", "category_Garden", "category_Grocery", "category_Meat", "category_Miscellaneous", "category_Natural Products", "category_Packaged Meat", "category_Pharmaceutical", "category_Prepared Food", "category_Restauarant", "category_Salads", "category_Seafood", "category_Skin & Hair Care", "category_Travel", "category_Vegetables (cut)"]:
    gdf = coup_df.groupby("coupon_id")[col].sum().reset_index()
    gdf.columns = ["coupon_id", "Coup_"+col+"_ohesum"]
    coup_df = pd.merge(coup_df, gdf, on="coupon_id", how="left")

coup_df = coup_df.drop(['category_Bakery', 'category_Dairy, Juices & Snacks', 'category_Flowers & Plants', 'category_Garden', 'category_Grocery', 'category_Meat', 'category_Miscellaneous', 'category_Natural Products', 'category_Packaged Meat', 'category_Pharmaceutical', 'category_Prepared Food', 'category_Restauarant', 'category_Salads', 'category_Seafood', 'category_Skin & Hair Care', 'category_Travel', 'category_Vegetables (cut)'],axis=1)
coup_df = coup_df.drop_duplicates()

for col in ["brand_type_Established", "brand_type_Local"]:
    gdf = coup_df.groupby("coupon_id")[col].sum().reset_index()
    gdf.columns = ["coupon_id", "Coup_"+col+"_ohesum"]
    coup_df = pd.merge(coup_df, gdf, on="coupon_id", how="left")

coup_df = coup_df.drop(['brand_type_Established', 'brand_type_Local'],axis=1)
coup_df = coup_df.drop_duplicates()

#del cust_df
#del gdf
dtype_coup = coup_df.dtypes.to_frame('dtypes').reset_index()
all_df = pd.merge(all_df, coup_df, on="coupon_id", how="left")

dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()
del coup_df

dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()
all_df = pd.merge(all_df, trans_df, on="customer_id", how="left")
dtype_all = all_df.dtypes.to_frame('dtypes').reset_index()
all_df_sam = all_df.head(1000)

del trans_df, trans_df_sam
del gdf
#from sklearn.utils import shuffle
#all_df= shuffle(all_df, random_state=12)

#Split Train and Test
train_df = all_df[all_df["train_set"]==1].reset_index(drop=True)
test_df = all_df[all_df["train_set"]==0].reset_index(drop=True)

gc.collect()
gc.collect()
n_splits = 10
gkf = model_selection.GroupKFold(n_splits=n_splits)

##
camp_unique=train_df.campaign_id.unique().tolist()

train_df[["customer_id"]] = train_df[["customer_id"]].apply(pd.to_numeric)
test_df[["customer_id"]] = test_df[["customer_id"]].apply(pd.to_numeric)

categ_map = {"Grocery":0, "Bakery":1, "Skin & Hair Care":2, "Pharmaceutical":3, "Seafood":4, "Packaged Meat":5, "Dairy, Juices & Snacks":6, "Natural Products":7, "Miscellaneous":8, "Prepared Food":9, "Meat":10, "Vegetables (cut)":11, "Travel":12, "Garden":13, "Flowers & Plants":14, "Salads":15, "Restauarant":16}
train_df["MCCoupcategory_c"] = train_df["MCCoupcategory_c"].map(categ_map)
test_df["MCCoupcategory_c"] = test_df["MCCoupcategory_c"].map(categ_map)

#dtype_test = test_df.dtypes.to_frame('dtypes').reset_index()
ID_y_col = ["id","redemption_status"]
#raw_col = [col for col in train_df.columns if col not in ID_y_col]
cols_to_leave = ["id", "redemption_status", "train_set", "start_date", "end_date"]
#cols_to_leave.extend(['s_wk', 's_mon', 's_yr', 'cus_str_shift1', 'e_dwk', 's_dwk', 'e_yr', 'e_mon', 'e_d', 'camp_id_count', 'camp_coup_uniq_p', 'camp_cus_uniq_p', 'cus_camp_shift1', 'diffn1_cus_str', 'diff1_cus_str', 'MCCoupcategory_c'])
cols_to_leave.extend(['MCCoupcategory_c'])
cols_to_leave.extend(['s_yr', 's_mon', 's_wk', 's_d', 's_dwk', 'e_yr', 'e_mon', 'e_wk', 'e_d', 'e_dwk', 'camp_coupon_idnunique', 'camp_customer_idnunique', 'camp_coup_unique_cust', 'camp_cus_unique_coup', 'camp_id_count', 'start_date_ord', 'end_date_ord', 'cus_str_shift1', 'cus_str_shift2', 'cus_str_shift3', 'cus_str_shiftn1', 'cus_str_shiftn2', 'cus_str_shiftn3', 'cus_camp_shift1', 'cus_camp_shift2', 'cus_camp_shift3', 'cus_camp_shiftn1', 'cus_camp_shiftn2', 'cus_camp_shiftn3', 'cus_coup_shift1', 'cus_coup_shift2', 'cus_coup_shift3', 'cus_coup_shiftn1', 'cus_coup_shiftn2', 'cus_coup_shiftn3', 'cus_end_shift1', 'cus_end_shift2', 'cus_end_shift3', 'cus_end_shiftn1', 'cus_end_shiftn2', 'cus_end_shiftn3'])

fe_col = [col for col in train_df.columns if col not in ID_y_col]
cols_to_use = []
cols_to_use = [i for i in fe_col if i not in cols_to_leave]
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = (train_df["redemption_status"]).values
test_id = test_df["id"].values
print(train_X.shape, test_X.shape)

#LightGBM function to define hyperparameters, early stopping; to get feature importance, loss
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep = 10, seed=2019, data_leaf=200):
    params = {}
    params["objective"] = "binary"
    params['metric'] = 'auc'
    params["max_depth"] = dep
    params["num_leaves"] = 10
    params["min_data_in_leaf"] = data_leaf
    params["learning_rate"] = 0.01
    params["bagging_fraction"] = 0.7
    params["feature_fraction"] = 0.45
#    params["feature_fraction_seed"] = seed
#    params["bagging_freq"] = 2
    params["bagging_seed"] = seed
#    params["scale_pos_weight"] = 20
    params["min_sum_hessian_in_leaf"] = 10
    params["lambda_l2"] = 15
    params["lambda_l1"] = 0.5
#    params["is_unbalance"] = True
    params["verbosity"] = -1
    num_rounds = 20000

    plst = list(params.items())
    lgtrain = lgb.Dataset(train_X, label=train_y)

    if test_y is not None:
        lgtest = lgb.Dataset(test_X, label=test_y)
        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtrain,lgtest], early_stopping_rounds=500, verbose_eval=500)
    else:
        lgtest = lgb.DMatrix(test_X)
        model = lgb.train(params, lgtrain, num_rounds)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    print("Features importance...")
    gain = model.feature_importance('gain')
    ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    ft.to_csv("am19_av_fimp_8.6.csv", index=False)
    print(ft.head(25))

    loss = 0
    if test_y is not None:
        loss = metrics.roc_auc_score(test_y, pred_test_y)
        print(loss)
        return model, loss, pred_test_y, pred_test_y2
    else:
        return model, loss, pred_test_y, pred_test_y2

print("Building model..")
start_time = time.time()
t = time.time()

cv_scores = []
pred_test_full = 0
pred_train = np.zeros(train_X.shape[0])
n_splits = 10
kf = model_selection.KFold(n_splits=n_splits, shuffle=False, random_state=7988)
#gkf = model_selection.GroupKFold(n_splits=n_splits)
#skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=2018)
model_name = "lgb"
for dev_index, val_index in kf.split(train_X, train_df["redemption_status"].values):
    dev_X, val_X = train_X.iloc[dev_index,:], train_X.iloc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val = 0
    pred_test = 0
    n_models = 0.

    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=10, seed=2019)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1

    pred_val /= n_models
    pred_test /= n_models

    loss = metrics.roc_auc_score(val_y, pred_val)

    pred_train[val_index] = pred_val
    pred_test_full += pred_test / n_splits
    cv_scores.append(loss)
#     break
print(np.mean(cv_scores))
f"Model took {time.time() - t: 6.2f} s"

#Submissions
out_df = pd.DataFrame({"id":test_id})
out_df["redemption_status"] = pred_test_full
out_df.to_csv("av_am19_out_8.6.csv", index=False)

