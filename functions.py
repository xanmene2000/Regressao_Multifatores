import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import requests
import datetime as dt
from urllib.parse import quote
