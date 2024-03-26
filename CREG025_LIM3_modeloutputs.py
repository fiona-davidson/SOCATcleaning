#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:35:19 2022

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
#import cmocean.cm as cmo
import pandas as pd
import xarray as xr
import datetime
from datetime import date, timedelta

path = '/gpfs/fs7/dfo/hpcmc/pfm/fid000/ANALYSIS/DATA/'
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

print(dir_list)

file_random = '/gpfs/fs7/dfo/hpcmc/pfm/fid000/RUN_DIR/Auto-restart/CREG025_LIM3/CREG025_LIM3-VFD003/CDF/CREG025_LIM3-VFD003_5d_grid_T_19931001-19931005.nc'
data_old = Dataset(file_random, "r", format = "NETCDF4")
print(type(data_old))
print(data_old.variables.keys())

# Put the data into NumPy arrays
lat_outfile = data_old.variables['nav_lat'][:]
long_outfile = data_old.variables['nav_lon'][:]
temp_outfile = data_old.variables['temp'][:]


# Run: CREG025_LIM3-VFD003

run_name = 'CREG025_LIM3-VFD003'

# dailyfiles = 1d, hourly files = 1h. daily = 1d
data_freq = '5d'

variable = 'temp'

start_date = '1993-10-01'
end_date = '1993-10-11' # Natasha puts the day after the last day she wants

upper_level = 0
lower_level = 1


# LOAD IN MESHMASK    
meshmask = Dataset('/gpfs/fs7/dfo/hpcmc/pfm/fid000/ANALYSIS/DATA/CREG025_mesh_mask.nc','r', format='NETCDF4') 


print(meshmask)

e1t = meshmask.variables['e1t'][:,:,:].squeeze() # y,x
e2t = meshmask.variables['e2t'][:,:,:].squeeze() # y,x
e3t = meshmask.variables['e3t_0'][:,:,:,:].squeeze() # z,y,x
e3t_1d = meshmask.variables['e3t_1d'][:,:].squeeze() # z
mbathy = meshmask.variables['mbathy'][:,:,:].squeeze() # y,x
meshz,meshy,meshx = e3t.shape
lon = meshmask.variables['nav_lon'][:,:]
lat = meshmask.variables['nav_lat'][:,:]
depth = meshmask.variables['gdept_1d'][:].squeeze() 

tmask = meshmask.variables['tmask'][:,:,:].squeeze()
tmask = np.array(tmask)
tmask = tmask[upper_level:lower_level,:,:]#.squeeze()

# Calculate grid area
grid_area = e1t*e2t

e3t = e3t[upper_level:lower_level,:,:]
e3t = np.ma.masked_array(e3t, tmask==0)
if upper_level < 40:
    hdep = np.sum(e3t, axis = 0)
    

# Load in data
# specify what file
carbonateChem = 0

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
filename = [run_name + '_' + data_freq + '_' + file_type + '_19931001-19931005.nc'] #'_201*.nc'
file = ''.join(run_dir + filename)

data = xr.open_mfdataset(file) # loads in multiple files


print(data)


d0 = datetime.datetime.strptime(start_date,'%Y-%m-%d')
d1 = datetime.datetime.strptime(end_date,'%Y-%m-%d')

date=d0
tt=0
T_zhw=np.zeros(len(data.time_counter))



###############################################################################
############ NATASHA'S CODE BELOW ######################

while date<=d1: # time loop calculate daily cause I keep running out of memory
    da = data.sel(time_counter = slice(date.strftime('%Y-%m-%d')),deptht=slice(upper_level,lower_level))
    if carbonateChem==0:
        T = da.variables[variable]#.squeeze()
    elif carbonateChem==1:
        # get carbon variables
        DIC = da.variables['dissolved_inorganic_carbon']#.squeeze()
        TA = da.variables['total_alkalinity']#.squeeze()
        rho0 = da.variables['sigma_theta']#.squeeze()
        # mask variables
        TA = np.ma.masked_array(TA, tmask==0)
        DIC = np.ma.masked_array(DIC, tmask==0)
        rho0 = np.ma.masked_array(rho0, tmask==0)
        if np.max(rho0)<100:
            rho0 = rho0+1000
        # convert units from mmol/m3 to umol/kg
        DIC = (DIC*1000)/rho0 # 1000umol/1mmol   / (kg/m3)
        TA = (TA*1000)/rho0 # 1000umol/1mmol   / (kg/m3)
        del(rho0)
        # calculate carbon var I want
        T,units=get_carbonvar(DIC,TA)

    lt,lz,ly,lx = T.shape

    # Calculate averages
    if upper_level < 40:
        ga=np.ma.masked_array(grid_area, tmask[upper_level,:,:]==0)
    elif upper_level == 40:
        ga=np.ma.masked_array(grid_area, tmask[0,:,:]==0)
        T_bot = np.zeros([ly, lx])
        bot_dz = np.zeros([ly, lx])

    #Ttt = T[tt,:,:,:].squeeze()
    Ttt = np.ma.masked_array(T, tmask==0)
    
    if upper_level < 40:
        # weight in the vertical
        if lz == 1:
            T_zw = Ttt
        else:
            T_zw = np.sum(Ttt[upper_level:lower_level,:,:]*e3t[upper_level:lower_level,:,:], axis = 0)/hdep
        # weight in the horizontal
        T_zw = T_zw.squeeze()
        T_zhw[tt] = np.sum(T_zw*ga)/np.sum(ga)
    elif upper_level == 40:
        for yy in range(0,ly):
            for xx in range(0,lx):
                T_bot[yy, xx] = Ttt[mbathy[yy, xx] - 1, yy, xx] # data in bottom cell
                bot_dz[yy,xx] = e3t[mbathy[yy, xx] - 1, yy, xx] # bottom cell thickness
        T_zhw[tt,0] = np.sum(T_bot*ga)/np.sum(ga)
        print('NOT VERTICALLY WEIGHTED YET')
    tt += 1
    date += timedelta(days=1)


# PLOT THE TIMESERIES
figurename=''.join([variable + 'TimeSeries_z' + str(upper_level) + '-' + str(lower_level) + '_' + start_date + '_' + end_date + '.png'])

fig = plt.figure(figsize = (4, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot(da['time_counter'], T_zhw)
ax.set_xlabel('date')
if variable in ['temp', 'salt']:
    ax.set_ylabel(variable + ' [' + units + ']')
    ax.set_title(variable + ' time series')
else:
    ax.set_ylabel(data[variable].long_name + ' [' + data[variable].units + ']')
    ax.set_title(data[variable].long_name + ' time series')
ax.grid(True)
plt.savefig(figurename)
plt.show()
plt.close()












