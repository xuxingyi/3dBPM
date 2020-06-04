# calculate the local wake field and balance phase
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import signal

# calculate the local wake field
LUT1 = np.load("LUT1.npy")
LUT2 = np.load("LUT2.npy")
T = np.load("T.npy")
LUT1 = LUT1.reshape((np.size(LUT1), ), order='F')
LUT2 = LUT2.reshape((np.size(LUT2), ), order='F')
LUT1 = LUT1 - np.mean(LUT1[-1-1000:])
LUT2 = LUT2 - np.mean(LUT2[-1-1000:])
Ind1 = np.argmin(LUT1)
tmp = LUT2[Ind1 - 2000:Ind1 + 2000]
Ind2 = np.argmin(tmp)
DeltaInd = 2000 - Ind2
LUT2 = np.roll(LUT2, DeltaInd)
del tmp
tmp = LUT1 + LUT2
WakeField1 = LUT1 - np.min(LUT1)*tmp/np.min(tmp)
WakeField2 = LUT2 - np.min(LUT2)*tmp/np.min(tmp)
WakeField = WakeField1 - WakeField2
NewLUT1 = LUT1 - WakeField1
NewLUT2 = LUT2 - WakeField2