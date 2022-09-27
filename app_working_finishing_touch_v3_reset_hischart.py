from dash import Dash, html, dcc, Input, Output, State, dash_table,ctx
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date
import pickle
import plotly.express as px

from app_analysis import data_preprocess,make_plot,historical_price_vol

import time
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np
import base64
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

######################################################################################################
### DATA
######################################################################################################

def parse_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return df

    else:
        return [{}]

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

clean_merged_data,clean_merged_data_agg_cy = data_preprocess(price_data,vol_data)


period_dict = {}
n_click_reset = 0

fig1 = {}
#fig1,_,_ = make_plot(clean_merged_data,'Overall',clean_merged_data.index.min(),clean_merged_data.index.max(),'CBOB')


# _,results,_ = make_plot(clean_merged_data,'Overall',clean_merged_data.index.min(),clean_merged_data.index.max(),'CBOB',n_click_reset)
# data = results.drop([0,1],axis = 0)
# data = data.to_dict('rows')
# columns =  [{"name": i, "id": i,} for i in (results.columns)]
# results_table = dash_table.DataTable(data=data,columns=columns)

results_table = []

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


        #############################################################




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


        html.Button(id='reset-button', n_clicks=0, children='Click to Reset App'),


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

        dbc.Row([
        
        dbc.Col(dcc.Upload(
        id='price-upload-data',
        children=html.Div(
            [
            'Price Data : Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
       

        },
        # Allow multiple files to be uploaded
        multiple=False
        )),
        

        dbc.Col(dcc.Upload(
        id='volume-upload-data',
        children=html.Div([
            'Volume Data : Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
      
        },
        # Allow multiple files to be uploaded
        multiple=False
        )),
        ]),

        html.Hr(),
        dcc.Graph(id = 'hischart', figure = fig1),
        dcc.Graph(id = 'chart', figure = fig1),
        # html.Div(id='display_ols', style={'whiteSpace': 'pre-wrap'}),
        html.Div(id="table1",children = results_table),
        

        

    ],
    style = CONTENT_STYLE
)

######################################################################################################
### LAYOUT
######################################################################################################

def serve_layout():

    return html.Div([dcc.Location(id="url"), sidebar, content])

app.layout = serve_layout()

######################################################################################################
### CALLBACKS
######################################################################################################

@app.callback(
    Output('hischart','figure'),
    [State('price-upload-data','contents'),
    State('price-upload-data','filename'),
    State('price-upload-data','last_modified'),
    Input('volume-upload-data','contents'),
    State('volume-upload-data','filename'),
    State('volume-upload-data','last_modified'),],
    prevent_initial_call=True,
)

def update_hischart(price_contents,price_filename,price_last_modfidied,volume_contents,volume_filename,volume_last_modified):
    if price_contents is not None:
        price_data = parse_data(price_contents, price_filename)

    if volume_contents is not None:
        vol_data = parse_data(volume_contents, volume_filename)




    _,clean_merged_data_agg_cy = data_preprocess(price_data,vol_data)

    fig = historical_price_vol(clean_merged_data_agg_cy)


    
    return fig


@app.callback(
    [Output('chart', 'figure'),
    Output('table1','children')],
    [Input('first-var', 'value'),
    Input('submit-button-state','n_clicks')],
    [State('price-upload-data','contents'),
    State('price-upload-data','filename'),
    State('price-upload-data','last_modified'),
    State('volume-upload-data','contents'),
    State('volume-upload-data','filename'),
    State('volume-upload-data','last_modified'),
    State('input-1-state-name','value'),
    State('my-date-picker-range1','start_date'),
    State('my-date-picker-range1','end_date')]
    ,
    Input('reset-button','n_clicks'),
    prevent_initial_call=True,
)

def update_plot(var1,n_clicks,price_contents,price_filename,price_last_modfidied,volume_contents,volume_filename,volume_last_modified,period,start_date,end_date,n_click_reset):
    if price_contents is not None:
        price_data = parse_data(price_contents, price_filename)

    if volume_contents is not None:
        vol_data = parse_data(volume_contents, volume_filename)

    clean_merged_data,_ = data_preprocess(price_data,vol_data)


    if 'reset-button' == ctx.triggered_id:
        n_click_reset = 1
    else:
        n_click_reset = 0

    
    fig,_,_ = make_plot(clean_merged_data,period,start_date,end_date,var1,n_click_reset)
    _,results,_ = make_plot(clean_merged_data,period,start_date,end_date,var1,n_click_reset)
    #results = results.round(4)

    data = results.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (results.columns)]
    return fig,dash_table.DataTable(data=data,columns=columns)

# @app.callback(
#     Output('chart', 'figure'),
#     Input('first-var', 'value'),
#     Input('submit-button-state','n_clicks'),
#     State('input-1-state-name','value'),
#     State('my-date-picker-range1','start_date'),
#     State('my-date-picker-range1','end_date'),
#     prevent_initial_call=True,
# )

# def update_plot(var1,n_clicks,period,start_date,end_date,):
#     fig,_,_ = make_plot(clean_merged_data,period,start_date,end_date,var1)
#     return fig

# @app.callback(
#     Output('table1', 'children'),
#     Input('first-var', 'value'),
#     Input('submit-button-state','n_clicks'),
#     State('input-1-state-name','value'),
#     State('my-date-picker-range1','start_date'),
#     State('my-date-picker-range1','end_date'),
#     prevent_initial_call=True,

# )

# def update_table(var1,n_clicks,period,start_date,end_date,):

#     _,results,_ = make_plot(clean_merged_data,period,start_date,end_date,var1)
#     #results = results.round(4)

#     data = results.to_dict('rows')
#     columns =  [{"name": i, "id": i,} for i in (results.columns)]
#     return dash_table.DataTable(data=data,columns=columns)

    




######################################################################################################
### RUN
######################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)