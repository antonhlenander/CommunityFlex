import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

subfolder = 'data/fullyearPV_singleDemand/'
demand_path = f'{subfolder}demandprofiles.csv'
prod_path = f'{subfolder}PV.csv'
price_path = f'{subfolder}elspotprices.csv'
cap_path = 'data/eval/caps.csv'

elafgift = 0.7630
tso = 0.049 + 0.061 + 0.0022
dso_radius_winter = [0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,0.6889,2.0666,2.0666,2.0666,2.0666,0.6889,0.6889,0.6889]
dso_radius_summer = [0.2296,0.2296,0.2296,0.2296,0.2296,0.2296,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.3444,0.8955,0.8955,0.8955,0.8955,0.3444,0.3444,0.3444]
vindstoed = 0.00375 + 0.000875 + 0.01

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
    # Drop the 'hour' column
    df.drop(columns=['hour'], inplace=True)
    # Return the dataframe
    cols = df.columns.tolist()
    cols = cols[:1] + cols[6:] + cols[1:6]
    df = df[cols]
    # Divide every column from the 3rd column onwards by 1000
    df.iloc[:] = df.iloc[:] / 1000
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
    #df = df[df['time'].dt.date == pd.to_datetime('2019-08-01').date()]
    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Copy the production column to create 14 identical columns
    for i in range(1, 15):
        df[f'H{i}'] = df['production']
    # Drop 'production column
    df.drop(columns=['production'], inplace=True)
    return df

def load_price_data(price_path):
    price_df = pd.read_csv(price_path, delimiter=";", header=0)
    # Drop all columns except 'SpotPriceDKK'
    price_df = price_df['SpotPriceDKK'].str.replace(',', '.').astype(float)
    # Divide every value by 1000 to convert to DKK/kWh
    price_df = price_df / 1000
    elafgift = 0.7630
    tso = 0.049 + 0.061 + 0.0022
    price_df = price_df + elafgift + tso
    return price_df

def load_cap_data(cap_path):
    cap_df = pd.read_csv(cap_path, delimiter=";", header=0)
    return cap_df

df = load_price_data(price_path)
print(df.head())
