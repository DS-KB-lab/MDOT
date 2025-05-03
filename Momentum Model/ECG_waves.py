import neurokit2 as nk
import numpy as np
import pandas as pd


ecg_signal = nk.data(dataset="ECGdata")
# Extract R-peaks locations
signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
                                         rpeaks, 
                                         sampling_rate=3000, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='all')

# Visualize P-peaks and T-peaks
signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
                                         rpeaks, 
                                         sampling_rate=3000, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='peaks')

# visualize T-wave boundaries
signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
                                         rpeaks, 
                                         sampling_rate=3000, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='bounds_T')
# Visualize P-wave boundaries
signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
                                         rpeaks, 
                                         sampling_rate=3000, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='bounds_P')
# Visualize R-wave boundaries
signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, 
                                         rpeaks, 
                                         sampling_rate=3000, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='bounds_R')