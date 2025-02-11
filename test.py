from agents import DSO
from datamanager import DataManager
import prepros as pp
import numpy as np
import matplotlib.pyplot as plt


subfolder = 'data/fullyearPV_singleDemand/'
demand_path = f'{subfolder}demandprofiles.csv'
prod_path = f'{subfolder}PV.csv'
price_path = f'{subfolder}elspotprices.csv'
cap_path = f'data/eval/caps.csv'
