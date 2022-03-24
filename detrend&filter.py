'''
Author: ChenHJ
Date: 2022-03-21 21:07:17
LastEditors: ChenHJ
LastEditTime: 2022-03-24 15:37:28
FilePath: /chenhj/0302code/detrend&filter.py
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

import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp

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
# %%
#   butterworth bandpass filter for 2-8yr
highfreq = 2.0*1/24
lowfreq = 2.0*1/(8.0*12.0)
b, a = signal.butter(8, [highfreq, lowfreq], btype="bandpass")
filtedData = signal.filtfilt(b, a, data, axis=0)
