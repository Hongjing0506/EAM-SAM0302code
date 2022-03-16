'''
Author: ChenHJ
Date: 2022-03-16 17:42:02
LastEditors: ChenHJ
LastEditTime: 2022-03-16 19:07:49
FilePath: /chenhj/0302code/circulation_reg.py
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
fpreCRU = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]

fprehis = xr.open_dataset(
    "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/pr/pr_Amon_ensemble_historical_gn_195001-201412.nc"
)
prehis = fprehis["pr"]

pr_his_path = "/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/pr"
g = os.walk(pr_his_path)
filepath = []
modelname_pr = []
for path, dir_list, file_name in g:
    for filename in file_name:
        if re.search("ensemble", filename) == None:
            filepath.append(os.path.join(path, filename))
            loc = ca.retrieve_allstrindex(filename, "_")
            modelname_pr.append(filename[loc[1]+1:loc[2]])
preds_his = xr.open_mfdataset(filepath, concat_dim="models", combine='nested')
prehis_ds = xr.DataArray(preds_his['pr'])
prehis_ds.coords["models"] = modelname_pr

fvERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc")
print(fvERA5)
# vERA5 = fvERA5[""]

fvhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc")
vhis = fvhis["va"]
# %%
