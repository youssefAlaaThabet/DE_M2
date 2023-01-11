import plotly.express as px 
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


def M4():

    df = pd.read_csv("/opt/airflow/data/2017_Accidents_UK.csv",dtype={'accident_index': 'str', 'accident_reference': 'str'})



    def histo2(df):
        fig= px.histogram(df, x='number_of_vehicles',y='day_of_week')
        return fig

    def trial(df):
        fig =px.scatter(df,x='number_of_vehicles',y='road_surface_conditions')
        return fig

    def histogram(df):
        fig = px.histogram(df,x="urban_or_rural_area")
        return fig

    def histo3(df):
        fig = px.pie(df, names='accident_severity', values='number_of_vehicles')
        return fig

    def histo4(df):
        result1 = df.groupby('light_conditions').accident_severity.value_counts()
        fig = px.density_heatmap(df, x='light_conditions', y='accident_severity')
        return fig


    # In[5]:


    app = Dash()
    app.layout = html.Div([
    
        html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),
        html.Br(),
        html.H1("2017_Accidents_UK dataset", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Relation between light condition and severness of accident", style={'text-align': 'center'}),
        dcc.Graph(figure=histo4(df)),
        html.Div(id='output_container', children=[]),
        html.Br(),
        html.H1("Relation between the number of vehicles in the accident and the surface conditions", style={'text-align': 'center'}),
        dcc.Graph(figure=trial(df)),
        html.Br(),
        html.Div(),
        html.H1("Number of vehicles got in accident on specific day", style={'text-align': 'center'}),
        dcc.Graph(figure=histo2(df)),
        html.Br(),
        html.Div(),
        html.H1("Relation between number of accidents and it's severness", style={'text-align': 'center'}),
        dcc.Graph(figure=histo3(df)),
        html.Br(),
        html.Div(),
        html.H1("Number of accidents occurred in rural or urban areas", style={'text-align': 'center'}),
        dcc.Graph(figure=histogram(df)),
        html.Div(id='output_container', children=[]),
        html.Br(),
    ])

    app.run_server(debug=False)



