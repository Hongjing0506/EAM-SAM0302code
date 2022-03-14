'''
Author: ChenHJ
Date: 2022-03-13 10:26:30
LastEditors: ChenHJ
LastEditTime: 2022-03-14 14:49:28
FilePath: /chenhj/0302code/precip_reg.py
Aim: 
Mission: 
'''
# %%
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
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
# %%
#   read the data file
fpreCRU = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc")
preCRU = fpreCRU["pre"]

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc")
preGPCP = fpreGPCP["precip"]

# %%
preCRU_India = preCRU.loc[:, 8:28, 70:86]
preCRU_EA = preCRU.loc[:, 36:42, 108:118]
preCRU_Japan = preCRU.loc[:, 31:36, 130:140]

preGPCP_India = preGPCP.loc[:, 8:28, 70:86]
preGPCP_EA = preGPCP.loc[:, 36:42, 108:118]
preGPCP_Japan = preGPCP.loc[:, 31:36, 130:140]
# %%
#   calculate area mean precipitation
preCRU_India_mean = ca.cal_lat_weighted_mean(preCRU_India).mean(dim="lon", skipna=True)
preCRU_EA_mean = ca.cal_lat_weighted_mean(preCRU_EA).mean(dim="lon", skipna=True)
preCRU_Japan_mean = ca.cal_lat_weighted_mean(preCRU_Japan).mean(dim="lon", skipna=True)

preGPCP_India_mean = ca.cal_lat_weighted_mean(preGPCP_India).mean(dim="lon", skipna=True)
preGPCP_EA_mean = ca.cal_lat_weighted_mean(preGPCP_EA).mean(dim="lon", skipna=True)
preGPCP_Japan_mean = ca.cal_lat_weighted_mean(preGPCP_Japan).mean(dim="lon", skipna=True)


# %%
