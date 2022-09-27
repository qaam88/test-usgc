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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    clean_merged_data = clean_merged_data.rename(columns={'PCBOB_TO_CBOB':'PCBOB_BASIS','PBOB_TO_RBOB':'PBOB_BASIS','JET_REGRADE':'JET_BASIS'})
    
    
    clean_merged_data_vol_list = clean_merged_data.columns[clean_merged_data.columns.str.contains('NetCPL')]
    
    
    for i in clean_merged_data_vol_list:
        clean_merged_data['K'+ i] = clean_merged_data[i]/1000
        
    clean_merged_data_vol_list = clean_merged_data.columns[clean_merged_data.columns.str.contains('NetCPL')]
        
    clean_merged_data_price_list = clean_merged_data.columns[clean_merged_data.columns.str.contains('_BASIS')]
    
    clean_merged_data_agg_cy_price = clean_merged_data.groupby('yr_cy')[clean_merged_data_price_list].mean()
    clean_merged_data_agg_cy_volume = clean_merged_data.groupby('yr_cy')[clean_merged_data_vol_list].sum()
    clean_merged_data_agg_cy = clean_merged_data_agg_cy_price.merge(clean_merged_data_agg_cy_volume,how='inner', on='yr_cy', sort=True)
    
    
    
    

    
    

    return clean_merged_data,clean_merged_data_agg_cy

clean_merged_data = data_preprocess(price_data,vol_data)

period_dict = {}


def make_plot(df,period,start_date1,end_date1,product,n_click_reset = 1):
    
    master_df = pd.DataFrame()
    analysis = []
    global period_dict
    period_dict[str(period)] = (start_date1,end_date1)
    if n_click_reset > 0 :
        period_dict = {}
        return {},pd.DataFrame(columns=["NetCPL_Sold_Vol_Mean","Durbin_Watson","NetCPL_Sold_Coeff_pvalue","NetCPL_Sold_Coeff_Mean","Coeff_Hi_95%_CI","Coeff_Lo_95%_CI","Coeff_Hi_90%_CI","Coeff_Lo_90%_CI","CPL_HOP_impact_mean","CPL_HOP_impact_95Hi","CPL_HOP_impact_95Lo"]),pd.DataFrame

    for key in period_dict:
        frame = []
        summer_df = df[period_dict[key][0]:period_dict[key][1]].query('CBOB_RVP == 9')
        winter_df = df[period_dict[key][0]:period_dict[key][1]].query('CBOB_RVP > 9')
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
#         product_summer_df.describe()

        product_winter_df['price_basis_stationary'] = product_winter_df['price_basis_average'].diff()
        product_winter_df['volume_stationary'] = product_winter_df['vol_sum'].diff()
        product_winter_df['Kvolume_stationary'] = product_winter_df['Kvol_sum'].diff()
        product_winter_df.dropna(inplace=True)
#         product_winter_df.describe()
        frame.append(product_summer_df)
        frame.append(product_winter_df)
        placeholder_df = pd.concat(frame, keys= [str(key) + "_summer", str(key) + "_winter"])
        name_list = [str(key)+' summer', str(key)+' winter']
        
        
        master_df = pd.concat([master_df,placeholder_df])
    

        for (item,name) in zip(frame,name_list):

            y = item['price_basis_stationary']
            x = item['Kvol_sum']

            # add constant for intercept
            x = sm.add_constant(x)

            # fitting model
            model = sm.OLS(y,x, missing='drop')
            ols_results = model.fit()

            kvol_means = item['Kvol_sum'].mean()
            pvals = ols_results.pvalues.Kvol_sum
            coeff = ols_results.params.Kvol_sum
            hi_coef = ols_results.conf_int(alpha = 0.05).loc['Kvol_sum', 0]
            low_coef = ols_results.conf_int(alpha = 0.05).loc['Kvol_sum', 1]
            hi_coef90 = ols_results.conf_int(alpha = 0.1).loc['Kvol_sum', 0]
            low_coef90 = ols_results.conf_int(alpha = 0.1).loc['Kvol_sum', 1]
            dw = durbin_watson(ols_results.wresid)
            CPL_HOP_impact_mean = kvol_means * coeff
            CPL_HOP_impact_95Hi = kvol_means * hi_coef
            CPL_HOP_impact_95Lo = kvol_means * low_coef
            date_range = str(period_dict[key][0])+' -> '+ str(period_dict[key][1])

            results_df = pd.DataFrame({"NetCPL_Sold_Coeff_pvalue":pvals,
                                    "NetCPL_Sold_Coeff_Mean":coeff,
                                    "Coeff_Hi_95%_CI":hi_coef,
                                    "Coeff_Lo_95%_CI":hi_coef,
                                    "Coeff_Hi_90%_CI":hi_coef90,
                                    "Coeff_Lo_90%_CI":low_coef90,
                                    "Durbin_Watson": dw,
                                    "NetCPL_Sold_Vol_Mean": kvol_means,
                                    "CPL_HOP_impact_mean": CPL_HOP_impact_mean,
                                    "CPL_HOP_impact_95Hi": CPL_HOP_impact_95Hi,
                                    "CPL_HOP_impact_95Lo": CPL_HOP_impact_95Lo,
                                    "Date_Range": date_range,
                                        },index = [name])

            #Reordering...
            results_df = results_df[["Date_Range","NetCPL_Sold_Vol_Mean","Durbin_Watson","NetCPL_Sold_Coeff_pvalue","NetCPL_Sold_Coeff_Mean","Coeff_Hi_95%_CI","Coeff_Lo_95%_CI","Coeff_Hi_90%_CI","Coeff_Lo_90%_CI","CPL_HOP_impact_mean","CPL_HOP_impact_95Hi","CPL_HOP_impact_95Lo"]]

            analysis.append(results_df)
            
    
        
#     frame = [product_summer_df,product_winter_df]
#     merged_df = pd.concat(frame,keys=["summer","winter"])
    
    fig= px.scatter(master_df, x='Kvol_sum', y='price_basis_stationary',trendline ='ols', color = master_df.index.get_level_values(0), height = 700)
    results = pd.concat(analysis)

    ########added line to handle duplicate tables########
    results = results.round(4)
    results = results.reset_index()
    results = results.drop_duplicates(subset = 'index')
    #####################################################

    return fig,results,master_df


def historical_price_vol(df):
    quote_ls = df.columns[df.columns.str.contains('_BASIS')]

    type = "category"   
    rangeselector = None
    df.index = df.index.astype('category')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    for q in quote_ls:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[q], name=str(q)+ "(cpg)"),
            secondary_y=True,)
    
        fig.add_trace(
            go.Bar(x=df.index, y=df['KNetCPL_'+ q[:-6]], name="CPL HoP Net Traded "+str(q[:-6])+" (KBBL/CY)"),
            secondary_y=False,
        )
    # Add figure title
    fig.update_layout(
        title_text='Historical USGC Spot Basis Vs EM HOP Net Traded Spot Volume'
    )

    # Set x-axis title
    titl = "Trading Cycle" 
    fig.update_xaxes(title_text=str(titl))

    # Set y-axes titles
    fig.update_yaxes(title_text="USGC HoP Net Traded (KBBL/CY)", secondary_y=False)
    fig.update_yaxes(title_text="USGC NYMEX Basis Differential (cpg)", secondary_y=True)
#     fig.update_layout(legend=dict(
#     orientation="h",
#     yanchor="bottom",
#     y=1,
#     xanchor="right",
#     x=1))
    
    fig.update_layout(
    xaxis=dict(
        rangeselector=rangeselector,
        
        rangeslider=dict(
            visible=True
        ),
        type=type
    ),
    
    height = 700,
    width = 1650,)
    
    return fig

    
    