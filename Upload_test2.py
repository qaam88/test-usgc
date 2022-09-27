import base64
import datetime
import io
import plotly.graph_objs as go
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from zipfile import ZipFile
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from plotly.offline import plot, iplot

import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
"graphBackground": "#F5F5F5",
"background": "#ffffff",
"text": "#000000"
}

app.layout = html.Div([
dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Drag and Drop or ',
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
        'margin': '10px'
    },
    # Allow multiple files to be uploaded
    multiple=True
),
#dcc.Graph(id='Mygraph'),
#dcc.Graph(id='Mygraph2'),
#dcc.Graph(id='Mygraph3'),
html.Div(id='output-data-upload')
])





def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),delimiter="	", keep_default_na=False, dtype=str)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

        return df


#@app.callback(Output('output-data-upload', 'children'),
#          [
#Input('upload-data', 'contents'),
#Input('upload-data', 'filename')
#])
#def update_table(contents, filename):
#    table = html.Div()

#    if contents:
#        m = [i for i, s in enumerate(filename) if '1aat' in s]
#        index = m[0]
#        #index = contents.index(root)
#        contents = contents[index]
#        filename = filename[index] 
#        #contents = contents[0]
#        #filename = filename[0]
#        df = parse_data(contents, filename)

#        table = html.Div([
#            html.H5(filename),
#            dash_table.DataTable(
#                data=df.to_dict('rows'),
#                columns=[{'name': i, 'id': i} for i in df.columns]
#            ),
#            html.Hr(),
#            html.Div('Raw Content'),
#            html.Pre(contents[0:200] + '...', style={
#                'whiteSpace': 'pre-wrap',
#                'wordBreak': 'break-all'
#            })
#        ])

#    return table

import itertools
#@app.callback(Output('output_uploaded', 'children'),
#              [Input('upload_prediction', 'contents')],
#              [State('upload_prediction', 'filename'),
#               State('upload_prediction', 'last_modified')])

#app.callback(Output('output-data-upload', 'children'),
          #
#nput('upload-data', 'contents'),
#nput('upload-data', 'filename')
#)
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents, filename, last_modified):
    table = html.Div()
    #for content, name, date in zip(contents, filename, last_modified):
    if contents:
        m = [i for i, s in enumerate(filename) ]# '1aat' in s]
        index = m[0]
#        #index = contents.index(root)
        contents  = contents[index]
        filename = filename[index]
    
        # the content needs to be split. It contains the type and the real content
        content_type, content_string = contents.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
        zip_obj = ZipFile(zip_str, 'r')

        dfs = {text_file.filename: pd.read_csv(zip_obj.open(text_file.filename))
               for text_file in zip_obj.infolist()
               if text_file.filename.endswith('.xls')}
        
        dfg = {}
        for text_file in zip_obj.infolist():
            if 'price_data_reg_conv' in text_file.filename:
                df_tmp = pd.read_csv(zip_obj.open(text_file.filename), delimiter='\t')
                df_tmp['filename'] = text_file.filename
                dfg[text_file.filename] = df_tmp.reset_index()
                
        df = pd.concat(dfg.values())
        
        table = html.Div([
            #tml.H5(filename),
            dash_table.DataTable(
                data=df.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in df.columns]
            ),
            html.Hr(),
            html.Div('Raw Content'),
            #tml.Pre(contents[0:200] + '...', style={
            #   'whiteSpace': 'pre-wrap',
            #   'wordBreak': 'break-all'
            #)
        ])

    return table

                


if __name__ == '__main__':
    app.run_server(debug=True)