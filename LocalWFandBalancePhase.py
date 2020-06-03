# calculate the local wake field and balance phase
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import signal

LUT1 = np.load("LUT1.npy")
LUT2 = np.load("LUT2.npy")
T = np.load("T.npy")
LUT1 = LUT1.reshape((np.size(LUT1), ) , order='F')
LUT2 = LUT2.reshape((np.size(LUT2), ) , order='F')





LUT1=reshape(LUT1,1,max(size(LUT1))*720);
LUT2=reshape(LUT2,1,max(size(LUT2))*720);
LUT1 = LUT1 - mean(LUT1(end-1000:end));
LUT2 = LUT2 - mean(LUT2(end-1000:end));
[Amp Ind1] = min(LUT1);
tmp = LUT2(Ind1-2000:Ind1+2000);
[Amp Ind2] = min(tmp);
DeltaInd = 2000 - Ind2;
LUT2 = circshift(LUT2,DeltaInd);
clear tmp
tmp = LUT1 + LUT2;
WakeField1 = LUT1 - min(LUT1)*tmp/min(tmp);
WakeField2 = LUT2 - min(LUT2)*tmp/min(tmp);
WakeField = WakeField1 - WakeField2;
NewLUT1 = LUT1 - WakeField1;
NewLUT2 = LUT2 - WakeField2;
