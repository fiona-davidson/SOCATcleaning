#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:21:26 2022

@author: fid000
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
import netCDF4 as nc
from netCDF4 import Dataset, date2num
import os
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#import cmocean.cm as cmo
import pandas as pd
import xarray as xr
import datetime
from datetime import date, timedelta

# Run: CREG025_LIM3-VFD003

run_name = 'CREG025_LIM3-VFD003'

# dailyfiles = 1d, hourly files = 1h. daily = 1d
data_freq = '5d'

variable = 'temp'

start_date = '1993-10-01'
end_date = '1993-10-11' # Natasha puts the day after the last day she wants

upper_level = 0
lower_level = 1

# Load in data
# specify what file

if variable in ['salt', 'temp']: # salinity and temperature
    print('Grabbing the grid_T file')
    file_type = 'grid_T'
    
elif variable in 'ssh': # sea surface height
    print('Grabbing the grid_T_2D file')
    file_type = 'grid_T_2D'
    
else:
    print('Cannot find variable, check code and spelling')
    
    
# Load data
run_dir = ['/home/fid000/WORK7/RUN_DIR/Auto-restart/CREG025_LIM3/' + run_name + '/CDF/']
filename = [run_name + '_' + data_freq + '_' + file_type + '_1993*.nc'] #'_201*.nc'
file = ''.join(run_dir + filename)

data = xr.open_mfdataset(file) # loads in multiple files


print(data)


d0 = datetime.datetime.strptime(start_date,'%Y-%m-%d')
d1 = datetime.datetime.strptime(end_date,'%Y-%m-%d')
date=d0
tt=0
T_zhw=np.zeros(len(data.time_counter))




da = data.sel(time_counter = slice(date.strftime('%Y-%m-%d')),deptht=slice(upper_level,lower_level), drop=True)    
T = da.variables[variable].squeeze()
lz,ly,lx = T.shape     

#print(T[0,:,:].shape)
    
# PLOT THE TIMESERIES
figurename=''.join([variable + 'TimeSeries_z' + str(upper_level) + '-' + str(lower_level) + '_' + start_date + '_' + end_date + '.png'])

fig = plt.figure(figsize = (4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot(da['time_counter'], T)
ax.set_xlabel('date')
if variable in ['temp', 'salt']:
    #ax.set_ylabel(variable + ' [' + units + ']')
    ax.set_title(variable + ' time series')
else:
    ax.set_ylabel(data[variable].long_name + ' [' + data[variable].units + ']')
    ax.set_title(data[variable].long_name + ' time series')
ax.grid(True)
plt.savefig(figurename)
plt.show()
plt.close()













