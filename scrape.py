import requests
import csv
from tqdm import tqdm
import time

############################################################################################################
# Datahandler which retrieves data from the EnergiDataService API and saves it to CSV files
############################################################################################################

start_day = '01'
start_month = '01'
start_year = 2023

end_day = '01'
end_month = '07'
end_year = 2023

start_date = f'{start_year}-{start_month}-{start_day}T00:00'
end_date = f'{end_year}-{end_month}-{end_day}T00:00'

sortby = 'HourUTC%20ASC'

# Set the output paths for data
subfolder = 'data/01-01-23_to_01-07-23/'

consumption_csv_file_path = f'{subfolder}consumption_data.csv'

production_csv_file_path = f'{subfolder}production_data.csv'
production_filter_param = '{"PriceArea":["DK2"]}'

price_csv_file_path = 'bigdata/price_data.csv'

############################################################################################################
# Make and save consumption dataset
############################################################################################################

data = []
offset = 0
limit = 100000  # Adjust the limit based on the API's maximum allowed value
total_rows_approx = 900*24*30*(int(end_month) - int(start_month))


while True:
    paginated_url = f'https://api.energidataservice.dk/dataset/PrivIndustryConsumptionHour?offset={offset}&limit={limit}&start={start_date}&end={end_date}&sort={sortby}'
    response = requests.get(paginated_url)

    if response.status_code == 200:
        result = response.json()
        records = result.get('records', [])
        
        if not records:
            break  # Exit the loop if no more records are returned
        
        data.extend(records)

        print(f"Retrieved data from {offset} to {offset+limit} out of {total_rows_approx} rows")

        offset += limit  # Increment the offset for the next batch
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        break

# Define the CSV headers
if data:
    headers = data[0].keys()

    # Sort data by HourDK, then by MunicipalityNo, then by HousingCategory, then by HeatingCategory
    data.sort(key=lambda x: (x.get('HourDK'), x.get('HousingCategory'), x.get('HeatingCategory'), x.get('MunicipalityNo')))

    # Write data to CSV file
    with open(consumption_csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in tqdm(data, desc="Writing data to file", unit="row"):
            writer.writerow(row)

    print(f"Data successfully saved to {consumption_csv_file_path}")
else:
    print("No data found in the result")

############################################################################################################
# Get and save production dataset
############################################################################################################

# Make a GET request to the API
response = requests.get(
    url = f'https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement?offset=0&filter={production_filter_param}&start={start_date}&end={end_date}&sort={sortby}'
)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()

    # Extract the data from the result
    data = result.get('records', [])

else:
    print(f"Failed to retrieve data: {response.status_code}")
    data = []

# Define the CSV headers
if data:
    headers = data[0].keys()

    # Write data to CSV file
    with open(production_csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in tqdm(data, desc="Writing data to file", unit="row"):
            writer.writerow(row)

    print(f"Data successfully saved to {production_csv_file_path}")
else:
    print("No data found in the result")

############################################################################################################
# Get and save price dataset
############################################################################################################

# Make a GET request to the API
response = requests.get(
    url = f'https://api.energidataservice.dk/dataset/Elspotprices?offset=0&filter={production_filter_param}&start={start_date}&end={end_date}&sort={sortby}'
)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()

    # Extract the data from the result
    data = result.get('records', [])

else:
    print(f"Failed to retrieve data: {response.status_code}")
    data = []

# Define the CSV headers
if data:
    headers = data[0].keys()

    # Write data to CSV file
    with open(price_csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in tqdm(data, desc="Writing data to file", unit="row"):
            writer.writerow(row)

    print(f"Data successfully saved to {price_csv_file_path}")
else:
    print("No data found in the result")