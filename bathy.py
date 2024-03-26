#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:48:50 2022

@author: fid000
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs

data = xr.open_dataset('/gpfs/fs7/dfo/hpcmc/pfm/fid000/ANALYSIS/DATA/CREG025_mesh_mask.nc')
data = data.isel(t=0)
tmask = data['tmask'].values
e3t = data['e3t_0'].values
lon = data['glamt'].values
lat = data['gphit'].values

bathy = np.sum(e3t * tmask, axis=0)
bathy = np.ma.masked_where(bathy == 0, bathy)

fig = plt.figure(figsize=(12, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.pcolormesh(lon, lat, bathy, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, zorder=50)
ax.add_feature(cfeature.COASTLINE, zorder=51)
cb = plt.colorbar(cs, shrink=0.3)

plt.show(cs)

