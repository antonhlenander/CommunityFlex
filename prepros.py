import pandas as pd
import matplotlib.pyplot as plt

subfolder = 'data/'
demand_path = f'{subfolder}demandprofiles.csv'
prod_path = f'{subfolder}PV.csv'
price_path = f'{subfolder}price_data.csv'

def load_demand_profile(demand_path):
    print('Loading demand profiles...')
    df = pd.read_csv(demand_path, delimiter=";", header=3, skiprows=[4,5])
    # Drop all irrelavant columns
    df = df[['Dwelling index', 'Time', 'Net dwelling electricity demand']]
    # Rename columns
    df.rename(columns={
        'Dwelling index': 'house_index',
        'Time': 'time',
        'Net dwelling electricity demand': 'demand'
    }, inplace=True)
    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format="%I:%M:%S %p")
    # Extract the hour from the 'time' column
    df['hour'] = df['time'].dt.hour
    # Group by the hour and get the mean demand for each hour
    df = df.groupby(['house_index', 'hour'])['demand'].mean().reset_index()
    # Convert 'house_index' to string and add 'H' in front
    df['house_index'] = 'H' + df['house_index'].astype(str)
    # Pivot the dataframe to have a demand column for each house_index
    df = df.pivot(index='hour', columns='house_index', values='demand').reset_index()
    # Return the dataframe
    return df


def load_production_data(prod_path):
    print('Loading production data...')
    df = pd.read_csv(prod_path, header=3)
    # Drop all irrelavant columns
    df = df[['time', 'electricity']]
    # Rename column
    df.rename(columns={
        'electricity': 'production'
    }, inplace=True)
    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'])
    # Filter rows to only include data from 2019-08-01
    df = df[df['time'].dt.date == pd.to_datetime('2019-08-01').date()]
    # Convert production to watts
    df['production'] = df['production'] * 1000
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    return df

def load_price_data(price_path):
    price_df = pd.read_csv(price_path)
    return price_df