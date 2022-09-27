import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

#reading price data

price_data = pd.read_csv('price_data_v2.csv')

#reading volume data

vol_data = pd.read_csv('CPL_net_traded_volume.csv')

# converting date to datetime format

def data_preprocess(price_data,vol_data):
    price_data['Date'] = pd.to_datetime(price_data['Date'])

    # converting date to datetime format

    vol_data['AGREE_DATE'] = pd.to_datetime(vol_data['AGREE_DATE'])

    vol_data_min_date = vol_data.AGREE_DATE[1]
    vol_data_max_date = vol_data.AGREE_DATE.max()

    #renaming date columne for merging

    vol_data = vol_data.rename(columns={'AGREE_DATE':'Date'})

    #merging the price and volume data

    merged_data = price_data.merge(vol_data, how='outer', on='Date', sort=True)


    #preprocessing merged data

    merged_data1 = (merged_data
                    .drop(merged_data.index[0:2])
                    .set_index('Date')
                   )

    merged_data1['CBOB_CYCLE']=merged_data1['PCBOB_CYCLE']
    merged_data1[merged_data1.isna().any(axis=1)]

    date_series_to_drop = (merged_data1[merged_data1['CBOB_BASIS']
                                        .isna()]
                           .index
                          )
    date_series_to_drop

    merged_data2 = merged_data1.drop(date_series_to_drop)


    merged_data2.isna().any()

    #filling the dates with empty volumnes with 0 indicating no trade

    clean_merged_data = merged_data2.fillna(0)

    clean_merged_data = clean_merged_data[(clean_merged_data.index>=vol_data_min_date)&(clean_merged_data.index<=vol_data_max_date)]
    clean_merged_data


    ##################################################################################
    ###################################################################################
    ##beginning of analysis of data aggregated by cycle
    ####################################################################################

    clean_merged_data

    clean_merged_data = clean_merged_data[clean_merged_data.index.dayofweek <= 4]
    clean_merged_data

    #remove US holidays - there is no trading on these dates
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    cal = calendar()
    holidays = cal.holidays(start=clean_merged_data.index.min(), end=clean_merged_data.index.max())
    clean_merged_data['Holiday'] = clean_merged_data.index.isin(holidays)
    clean_merged_data = clean_merged_data[clean_merged_data['Holiday'] == False]
    clean_merged_data = clean_merged_data.drop(['Holiday'], axis=1)

    clean_merged_data = clean_merged_data.dropna()
    clean_merged_data

    ############################################################
    ###### Aggregate by Cycle and fix year inconsistency #######
    ############################################################

    # create field to identify year and trading cycle
    clean_merged_data['year'] = clean_merged_data.index.year
    clean_merged_data['month'] = clean_merged_data.index.month
    def year_adj(row):
        if row['month'] >10 and row['CBOB_CYCLE'] <15:
            new_yr = row['year']+1
        else:
            new_yr = row['year'] 
        return new_yr

    clean_merged_data['year'] = clean_merged_data.apply(lambda row: year_adj(row), axis=1)
    clean_merged_data['yr_cy'] = clean_merged_data['year'].astype('int32').astype(str)+"_"+clean_merged_data['CBOB_CYCLE'].astype('int32').astype(str).str.zfill(2)
    clean_merged_data

    clean_merged_data['KNetCPL_CBOB'] = clean_merged_data['NetCPL_CBOB']/1000
    clean_merged_data['KNetCPL_JET'] = clean_merged_data['NetCPL_JET']/1000
    clean_merged_data['KNetCPL_PBOB'] = clean_merged_data['NetCPL_PBOB']/1000
    clean_merged_data['KNetCPL_PCBOB'] = clean_merged_data['NetCPL_PCBOB']/1000
    clean_merged_data['KNetCPL_RBOB'] = clean_merged_data['NetCPL_RBOB']/1000
    clean_merged_data['KNetCPL_ULSD'] = clean_merged_data['NetCPL_ULSD']/1000
    clean_merged_data = clean_merged_data.rename(columns={'PCBOB_TO_CBOB':'PCBOB_BASIS','PBOB_TO_RBOB':'PBOB_BASIS','JET_REGRADE':'JET_BASIS'})

    return clean_merged_data

clean_merged_data = data_preprocess(price_data,vol_data)

def make_plot(df,start_date,end_date,product):
    summer_df = df[start_date:end_date].query('CBOB_RVP == 9')
    winter_df = df[start_date:end_date].query('CBOB_RVP > 9')
    
    product_summer_df = (summer_df
                            .groupby('yr_cy')
                            .agg(price_basis_average=(str(product) +'_BASIS','mean'),vol_sum = ('NetCPL_'+str(product),'sum'),Kvol_sum =('KNetCPL_'+str(product),'sum'))
                           )
    product_winter_df = (winter_df
                            .groupby('yr_cy')
                            .agg(price_basis_average=(str(product) +'_BASIS','mean'),vol_sum = ('NetCPL_'+str(product),'sum'),Kvol_sum =('KNetCPL_'+str(product),'sum'))
                           )
    
    product_summer_df['price_basis_stationary'] = product_summer_df['price_basis_average'].diff()
    product_summer_df['volume_stationary'] = product_summer_df['vol_sum'].diff()
    product_summer_df['Kvolume_stationary'] = product_summer_df['Kvol_sum'].diff()
    product_summer_df.dropna(inplace=True)
    product_summer_df.describe()
    
    product_winter_df['price_basis_stationary'] = product_winter_df['price_basis_average'].diff()
    product_winter_df['volume_stationary'] = product_winter_df['vol_sum'].diff()
    product_winter_df['Kvolume_stationary'] = product_winter_df['Kvol_sum'].diff()
    product_winter_df.dropna(inplace=True)
    product_winter_df.describe()
    
    frame = [product_summer_df,product_winter_df]
    merged_df = pd.concat(frame,keys=["summer","winter"])
    
    fig= px.scatter(merged_df, x='Kvol_sum', y='price_basis_stationary',trendline ='ols', color = merged_df.index.get_level_values(0), height = 800)
    
    results = px.get_trendline_results(fig)

    
    return fig,results

    
    