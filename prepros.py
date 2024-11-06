import pandas as pd

subfolder = 'data/01-01-23_to_01-07-23/'
cons_path = f'{subfolder}consumption_data.csv'
prod_path = f'{subfolder}production_data.csv'
price_path = f'{subfolder}price_data.csv'

# Modification of consumption data is done in the following steps:
# We are only interested in the complete consumption data for DK2, so all municipalities with a number above 410 are removed
# For each HousingCategory the consumption is split into two HeatingCategories (To measure electricity gone to heating)
# We sum the consumption for both categories and across all municipalities, 
# such that we get the total consumption for each HousingCategory for all of DK2

def load_consumption_data(cons_path):
    print('Loading consumption data...')
    cons_df = pd.read_csv(cons_path)
    # Filter out rows where 'MunicipalityNo' is above 410 such that we only get data from DK2
    cons_df = cons_df[cons_df['MunicipalityNo'] <= 410]
    # Sum the consumption values for each HousingCategory
    cons_df['ConsumptionkWh'] = cons_df.groupby(['HousingCategory', 'HourDK'])['ConsumptionkWh'].transform('sum')
    # Drop duplicates based on 'HousingCategory' and keep the first occurrence
    cons_df = cons_df.drop_duplicates(subset=['HourDK', 'HousingCategory']).reset_index(drop=True)
    # Drop the 'HeatingCategory' and 'HourUTC' column
    cons_df = cons_df.drop(columns=['HeatingCategory', 'HourUTC'])
    # Rename all values in column 'MunicipalityNo' to 'DK2' for good looks
    cons_df['MunicipalityNo'] = 'DK2'
    # Pivot the table such that 'HourDK' is the index and 'HousingCategory' is the columns
    cons_df = cons_df.pivot(index='HourDK', columns='HousingCategory', values='ConsumptionkWh')
    # Divide all columns by 1000 to convert kWh to MWh
    cons_df = cons_df / 1000
    return cons_df

# Modification of production data is done in the following steps:
# Filtering of price area is done in scraper
# We sum the production values for each power plant type

def load_production_data(prod_path):
    print('Loading production data...')
    prod_df = pd.read_csv(prod_path)
    # Extract the columns that will be used as index columns
    index_columns = prod_df.iloc[:, :3].copy()
    # Sum offshore wind power production in one column
    prod_df['OffshoreWind'] = prod_df.pop('OffshoreWindGe100MW_MWh') + prod_df.pop('OffshoreWindLt100MW_MWh')
    # Sum onshore wind power production in one column
    prod_df['OnshoreWind'] = prod_df.pop('OnshoreWindGe50kW_MWh') + prod_df.pop('OnshoreWindLt50kW_MWh')
    # Sum solar power production in one column, subtract self consumption
    prod_df['SolarPower'] = (
        prod_df.pop('SolarPowerGe40kW_MWh') + 
        prod_df.pop('SolarPowerLt10kW_MWh') + 
        prod_df.pop('SolarPowerGe10Lt40kW_MWh') - 
        prod_df.pop('SolarPowerSelfConMWh')
    )
    # Rename central power column
    prod_df.rename(columns={'CentralPowerMWh': 'CentralPower'}, inplace=True)
    # Sum local power plants + unknown production
    prod_df['LocalPower'] = prod_df.pop('LocalPowerMWh') + prod_df.pop('UnknownProdMWh')
    # Sum commercial power plants (waste incineration, etc.), subtract self consumption
    prod_df['CommercialPower'] = prod_df.pop('CommercialPowerMWh') - prod_df.pop('LocalPowerSelfConMWh')
    # Drop columns that are not needed NOTE: If DK1 is included in the data, this will need to be changed
    prod_df = prod_df.drop(columns=['HourUTC', 'HydroPowerMWh', 'ExchangeNO_MWh', 'ExchangeNL_MWh', 'ExchangeGB_MWh'])
    # TODO: Compute loss per timestep?
    return prod_df

def load_price_data(price_path):
    price_df = pd.read_csv(price_path)
    return price_df