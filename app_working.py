from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date
import pickle
import plotly.express as px

from app_analysis import data_preprocess,make_plot

import time
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np

######################################################################################################
### DATA
######################################################################################################

# Load data
#reading price data

price_data = pd.read_csv('price_data_v2.csv')

#reading volume data

vol_data = pd.read_csv('CPL_net_traded_volume.csv')

# def create_plot(df, columns):
#     fig = px.line(df, x=df.index.get_level_values(1), y=df[columns], color = df.index.get_level_values(0) ,facet_col = df.index.get_level_values(0),labels = dict(x="Date"))
#     fig.update_xaxes(matches=None)
#     return fig

# def create_ranked_plot(df, rank):
#     fig2 = px.line(df, x=df.index.get_level_values(1),y=df[rank], color = df.index.get_level_values(0),facet_col = df.index.get_level_values(0),labels = dict(x="Date"),height=300,width=1500)
#     fig2.update_xaxes(matches=None)
#     return fig2

# output = []
# #here you can define your logic on how many times you want to loop
# for i in rank: 
#      output.append(dcc.Graph(id='example-graph'+str(i),figure= create_ranked_plot(df, i)))

clean_merged_data = data_preprocess(price_data,vol_data)

period_dict = {}


######################################################################################################
### STYLE
######################################################################################################

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

######################################################################################################
### SIDEBAR
######################################################################################################

sidebar = html.Div(
    [
        html.H2("USGC Spot Market Price Analysis", className="display-4"),
        html.Hr(),
        #############################################################
        # html.H5("Product", className="display-4"),
        html.P(
            "Analyzing the relationship between price basis and the volume traded by XOM for various products", className="lead"
        ),
        html.Div('Select the product you are interested in'),
        dcc.Dropdown(
            ['CBOB','RBOB','PBOB','PCBOB','JET','ULSD'],
            'CBOB',
            id = 'first-var'
        ),
        
        html.Div('Please define the name of the period that you are interested in and the dates and click submit'),
        dcc.Input(id='input-1-state-name', type='text', value = 'Overall',placeholder='Overall'),
        
        dcc.DatePickerRange(
            id='my-date-picker-range1',
            min_date_allowed=clean_merged_data.index.min(),
            max_date_allowed=clean_merged_data.index.max(),
            initial_visible_month=date(2017, 12, 5),
            clearable=True,
            # start_date =date(2017,12,8),
            # end_date=date(2020, 2, 15)
        ),
        # html.Br,
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

        html.A(html.Button('Refresh App'),href='/'),


        # dcc.DatePickerRange(
        #     id='my-end-date',
        #     min_date_allowed=clean_merged_data.index.min(),
        #     max_date_allowed=clean_merged_data.index.max(),
        #     initial_visible_month=date(2017, 8, 5),
        #     end_date=date(2020, 2, 15)
        # ),



    ],
    style = SIDEBAR_STYLE,
)

######################################################################################################
### CONTENT
######################################################################################################

content = html.Div(
    [

        dcc.Graph(id = 'chart'),
        # html.Div(id='display_ols', style={'whiteSpace': 'pre-wrap'}),
        html.Div(id="table1"),
        

        

    ],
    style = CONTENT_STYLE
)

######################################################################################################
### LAYOUT
######################################################################################################

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

######################################################################################################
### CALLBACKS
######################################################################################################

@app.callback(
    Output('chart', 'figure'),
    Input('first-var', 'value'),
    Input('submit-button-state','n_clicks'),
    State('input-1-state-name','value'),
    State('my-date-picker-range1','start_date'),
    State('my-date-picker-range1','end_date'),
)

def update_plot(var1,n_clicks,period,start_date,end_date,):

    fig,_,_ = make_plot(clean_merged_data,period,start_date,end_date,var1)
    return fig

@app.callback(
    Output('table1', 'children'),
    Input('first-var', 'value'),
    Input('submit-button-state','n_clicks'),
    State('input-1-state-name','value'),
    State('my-date-picker-range1','start_date'),
    State('my-date-picker-range1','end_date'),

)

def update_table(var1,n_clicks,period,start_date,end_date,):

    _,results,_ = make_plot(clean_merged_data,period,start_date,end_date,var1)
    #results = results.round(4)

    data = results.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (results.columns)]
    return dash_table.DataTable(data=data,columns=columns)




######################################################################################################
### RUN
######################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)