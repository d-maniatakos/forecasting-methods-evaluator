from dataloader import DataLoader
from evaluation import Evaluation
from time_series import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt


time_series_list = DataLoader().time_series_list
Evaluation([time_series_list[0]]).evaluate()
