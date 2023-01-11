from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import sys
sys.path.append("/opt/airflow/data/")
from Data_Engineering_M1 import M1
from Data_Engineering_M1 import M2
from Data_Engineering_M1 import M3
from Viz_with_dash import M4

import matplotlib.pyplot as plt
import plotly.express as px 
import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

dataset = '2017_Accidents_UK.csv'

def extract_clean(filename):
    df = pd.read_csv(filename)
    df = clean_missing(df)
    df.to_csv('/opt/airflow/data/titanic_clean.csv',index=False)
    print('loaded after cleaning succesfully')
def encode_load(filename):
    df = pd.read_csv(filename)
    df = encoding(df)
    try:
        df.to_csv('/opt/airflow/data/titanic_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')
def clean_missing(df):
    df = impute_mean(df,'Age')
    df = impute_arbitrary(df,'Cabin','Missing')
    df = cca(df,'Embarked')
    return df
def impute_arbitrary(df,col,arbitrary_value):
    df[col] = df[col].fillna(arbitrary_value)
    return df
def impute_mean(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df
def impute_median(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df
def cca(df,col):
    return df.dropna(subset=[col])
def encoding(df):
    df = one_hot_encoding(df,'Embarked')
    df = label_encoding(df,'Cabin')
    return df
def one_hot_encoding(df,col):
    to_encode = df[[col]]
    encoded = pd.get_dummies(to_encode)
    df = pd.concat([df,encoded],axis=1)
    return df
def label_encoding(df,col):
    df[col] = preprocessing.LabelEncoder().fit_transform(df[col])
    return df
def load_to_csv(df,filename):
    df.to_csv(filename,index=False)



def load_to_postgres(filename1,filename2): 
    df = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/titanic')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_2017',con = engine,if_exists='replace')
    df2.to_sql(name = 'lookup_table',con = engine,if_exists='replace')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'titanic_etl_pipeline',
    default_args=default_args,
    description='titanic etl pipeline',
)
with DAG(
    dag_id = 'titanic_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['titanic-pipeline'],
)as dag:
    extract_clean_task= PythonOperator(
        task_id = 'extract_dataset',
        python_callable = M1,
        op_kwargs={
            "filename": '/opt/airflow/data/2017_Accidents_UK.csv'
        },
    )
    extract_figures_and_plots= PythonOperator(
         task_id = 'extract_figures_and_plots',
         python_callable = M2,
         op_kwargs={
             "filename": "/opt/airflow/data/final.csv"
         },
     )
    load_to_postgres_task=PythonOperator(
         task_id = 'load_to_postgres',
         python_callable = load_to_postgres,
         op_kwargs={"filename1": "/opt/airflow/data/final.csv","filename2": "/opt/airflow/data/lookup.csv"
         },
     )
    create_dashboard_task= PythonOperator(
          task_id = 'create_dashboard_task',
          python_callable = M4,
          op_kwargs={
              "filename": "/opt/airflow/data/2017_Accidents_UK.csv"
          },
      )
    


    extract_clean_task >> extract_figures_and_plots >> load_to_postgres_task >> create_dashboard_task
    



