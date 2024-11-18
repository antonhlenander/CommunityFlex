import pandas as pd
import matplotlib.pyplot as plt

subfolder = 'data/fullyearPV_singleDemand/'
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
    df['time'] = pd.to_datetime(df['time'], format="%I.%M.%S %p")
    # Extract the hour from the 'time' column
    df['hour'] = df['time'].dt.hour
    # Group by the hour and get the mean demand for each hour
    df = df.groupby(['house_index', 'hour'])['demand'].mean().reset_index()
    # Convert 'house_index' to string and add 'H' in front
    df['house_index'] = 'H' + df['house_index'].astype(str)
    # Pivot the dataframe to have a demand column for each house_index
    df = df.pivot(index='hour', columns='house_index', values='demand').reset_index()
    # Return the dataframe
    # Divide every column from the 3rd column onwards by 1000*60 to convert to kWh
    df.iloc[:, 1:] = df.iloc[:, 1:] / 1000
    return df

# def load_demand_profile2(demand_path):
#     #Loading generated profiles from CREST
#     df = pd.read_csv(demand_path, header=3, delimiter=';',skiprows=[4,5])
#     x = df |> @filter(_.var"Dwelling index" == 1) |> DataFrame
#     all_demands = zeros(24,N_consumers)
#     for j in 1:N_consumer
#         N = @from i in df begin
#                 @where i.var"Dwelling index" == j
#                 @select {time = i.Time, demand = i.var"Net dwelling electricity demand"}
#                 @collect DataFrame
#         end
#         N.time = Time.(N.time, "HH.MM.SS p")
#         hourly_demand = zeros(24)
#         for i in 0:23
#                 D = N |> @filter(Hour.(_.time) == Hour(i)) |> DataFrame
#                 hourly_demand[i+1] = sum(D[!,"demand"]./(1000*60))
#         end
#         all_demands[:,j] = hourly_demand
#     end
#     Demand = all_demands


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


    # Generation of solar PV systems and corresponding batteries (this should be random at some point)
    #Random.seed!(1234)
    #max_cap = rand(0:3,N_consumers)
    # Define production capacity for each house
    max_cap = [2, 1, 2, 3, 1, 1, 2, 3, 3, 1, 2, 3, 1, 1]
    for i in range(1, 15):
        # Multiply all columns by the max capacity for each
        df[f'H{i}'] = df[f'H{i}'] * max_cap[i-1]

    # Create a dataframe to store the battery capacities
    batt_cap_df = pd.DataFrame()
    for i in range(1, 15):
        batt_cap_df[f'H{i}'] = max_cap[i-1]*5

    return df, batt_cap_df

def load_price_data(price_path):
    price_df = pd.read_csv(price_path)
    return price_df