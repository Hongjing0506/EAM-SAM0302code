'''
Author: ChenHJ
Date: 2022-06-05 17:39:45
LastEditors: ChenHJ
LastEditTime: 2022-06-06 17:08:06
FilePath: /chenhj/0302code/test_Rossby_wave_source.py
Aim: 
Mission: 
'''
#%%
from mailbox import _PartialFile
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
import sys

sys.path.append("/home/ys17-23/chenhj/self_def/")
import plot as sepl
import cal as ca
import pandas as pd
from importlib import reload

import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind
import skill_metrics as sm
from brokenaxes import brokenaxes

reload(sepl)

# sd.path.append("/home/ys17-23/chenhj/1201code/self_def.py")

cdo = Cdo()

# for plot
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
from scipy import signal
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
#%%
#   read the data from ERA5 and calculate bandpass filter
fvERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc")
vERA5 = fvERA5["v"]
vERA5_fil = ca.butterworth_filter(vERA5, 8, 2*12, 9*12, "bandpass")
vERA5_MAM = ca.p_time(vERA5_fil, 3, 5, True).sel(level=250.0)

fuERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc")
uERA5 = fuERA5["u"]
uERA5_fil = ca.butterworth_filter(uERA5, 8, 2*12, 9*12, "bandpass")
uERA5_MAM = ca.p_time(uERA5_fil, 3, 5, True).sel(level=250.0)
# %%
#   rearange the coordinate of vERA5_MAM into -180°-180°
vERA5_MAM = ca.filplonlat(vERA5_MAM)
uERA5_MAM = ca.filplonlat(uERA5_MAM)
EOF_area_N = 75.0
EOF_area_S = 35.0
EOF_area_E = 120.0
EOF_area_W = -80.0

lat = vERA5_MAM.coords["lat"]
lon = vERA5_MAM.coords["lon"]
lat_EOF_range = lat[(lat >= EOF_area_S) & (lat <= EOF_area_N)]
lon_EOF_range = lon[(lon >= EOF_area_W) & (lon <= EOF_area_E)]
vERA5_MAM_EOF_area_std = ca.standardize(vERA5_MAM.sel(lat=lat_EOF_range, lon=lon_EOF_range))
vERA5_MAM_EOF, vERA5_MAM_pc1, vERA5_MAM_pcC = ca.eof_analyse(vERA5_MAM_EOF_area_std.data, lat_EOF_range.data, 1)
vERA5_MAM_EOF1 = vERA5_MAM_EOF[0,:,:]
vERA5_MAM_pc1 = np.squeeze(vERA5_MAM_pc1,axis=1)
# %%
#   calculate the meridional wind regress coefficients onto pc1
pc1_vERA5_slope, pc1_vERA5_intercept, pc1_vERA5_rvalue, pc1_vERA5_pvalue, pc1_vERA5_hypothesis = ca.dim_linregress(vERA5_MAM_pc1, vERA5_MAM)
pc1_uERA5_slope, pc1_uERA5_intercept, pc1_uERA5_rvalue, pc1_uERA5_pvalue, pc1_uERA5_hypothesis = ca.dim_linregress(vERA5_MAM_pc1, uERA5_MAM)

# %%
#   calculate the irrotational(divergence) wind