'''
Author: ChenHJ
Date: 2022-03-16 17:42:02
LastEditors: ChenHJ
LastEditTime: 2022-03-16 19:31:31
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
vERA5 = fvERA5["v"]

fvhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/va/va_Amon_ensemble_historical_gn_195001-201412.nc")
vhis = fvhis["va"]

fuERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc")
uERA5 = fuERA5["u"]

fuhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/ua/ua_Amon_ensemble_historical_gn_195001-201412.nc")
uhis = fuhis["ua"]

fqERA5 = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"]

fqhis = xr.open_dataset("/home/ys17-23/chenhj/SAM_EAM_data/CMIP6/historical/hus/hus_Amon_ensemble_historical_gn_195001-201412.nc")
qhis = fqhis["hus"]
# %%
#   calculate the meridional water vapor transport
#   select the level
uERA5_ver_Indian_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
uERA5_ver_EA_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

vERA5_ver_Indian_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
vERA5_ver_EA_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]

qERA5_ver_Indian_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 8:28, 70:86]
qERA5_ver_EA_JJA = ca.p_time(qERA5, 6, 8, True).loc[:, 100.0:, 36:42, 108:118]


uhis_ver_Indian_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
uhis_ver_EA_JJA = ca.p_time(uhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

vhis_ver_Indian_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
vhis_ver_EA_JJA = ca.p_time(vhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]

qhis_ver_Indian_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 8:28, 70:86]
qhis_ver_EA_JJA = ca.p_time(qhis, 6, 8, True).loc[:, :10000.0, 36:42, 108:118]
# %%
#   calculate the area mean
