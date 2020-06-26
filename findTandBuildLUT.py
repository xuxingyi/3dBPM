import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import signal
import h5py


# Get the right RF frequency information from BPM signal
f = h5py.File(
        r'20200220_213mA_AC_inject_6.h5',
        'r')
BPM1 = f['Waveforms']['Channel 1']['Channel 1Data'][()].astype("int32")

BPM3 = f['Waveforms']['Channel 3']['Channel 3Data'][()].astype("int32")

# define the peak index of the first bunch
flagForBaseline = False
PeakIndex = 0
for i in range(np.size(BPM1) // 100):
    if(flagForBaseline == False):
        if((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) < 4000):
            flagForBaseline = True
    elif((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) > 4000):
        for k in range(270):
            if((BPM1[(i - 1) * 100 + k + 10] == min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (max(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000):
                PeakIndex = (i - 1) * 100 + k + 10
                break
        break

PeakIndex1 = 0
flagForBaseline = False
for i in range(100,np.size(BPM1) // 100):
    if(flagForBaseline == False):
        if((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) < 4000):
            flagForBaseline = True
    elif((max(BPM1[i * 100:(i + 1) * 100]) - min(BPM1[i * 100:(i + 1) * 100])) > 4000):
        for k in range(270):
            if((BPM1[(i - 1) * 100 + k + 10] == min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) and (max(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20]) - min(BPM1[(i - 1) * 100 + k:(i - 1) * 100 + k + 20])) > 4000):
                PeakIndex1 = (i - 1) * 100 + k + 10
                break
        break

BaselineIndex = np.arange(PeakIndex1 - 1288, PeakIndex1 - 688)
#Filling = np.arange(499)


# find the right T using pickup #1
# process BPM1 signal
# copy raw data, include the previous 2 small bunches

Data = BPM1[PeakIndex - 10:].copy()
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline


# define the basic number
HarmonicNum = 720
# initial bucket size value, sampling rate 20GHz, 40*50ps = 2ns
T = 40
BunchSize = 40
# define which bunch will be processed
BunchIndex = 0
# define the data index for the first bunch and the dirst turn
DataIndexStart = BunchIndex * T

# calculate T value by counting the bunch numbers
turnNumT = np.size(Data) // 28000000
turnNum1 = np.array([2, 5, 10, 20, 100, 200, 500])
turnNum2 = np.arange(1, turnNumT + 1)
ScanTurnNum = np.hstack((turnNum1, turnNum2 * 1000))
N = len(ScanTurnNum)
BunchDataFirst = np.zeros(BunchSize, dtype="float64")
BunchDataEnd = np.zeros(BunchSize, dtype="float64")
for j in range(N):
    TurnNum = ScanTurnNum[j]
    DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                          T).astype("int32") + np.floor(BunchIndex * T).astype("int32")
    DataIndexE = DataIndexS + BunchSize
    # collect specified bunch data together
    BunchDataFirst = Data[DataIndexS[0]:DataIndexE[0]]
    BunchDataEnd = Data[DataIndexS[TurnNum - 1]:DataIndexE[TurnNum - 1]]
    # find the peak index for each turn of this bunch
    IndEnd = np.argmin(BunchDataEnd)
    IndFirst = np.argmin(BunchDataFirst)
    Peak1Index = IndFirst + DataIndexS[0] - 1
    PeakEndIndex = IndEnd + DataIndexS[-1] - 1
    # calculate the T using peak index
    T = (PeakEndIndex - Peak1Index) / 720 / (TurnNum - 1)

# build LUT of all bunches
TurnSize = np.floor(T * 720).astype("int32")
TurnNum = np.floor(len(Data) / 720 / T).astype("int32") - 1
# collect the all bunches data together using the new T value
DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                      T).astype("int32")
DataIndexE = DataIndexS + TurnSize
TurnData = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
TurnTime = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
for i in range(TurnNum):
    TurnData[:, i] = Data[DataIndexS[i]:DataIndexE[i]].reshape(TurnSize,)
    TurnTime[:, i] = np.arange(DataIndexS[i], DataIndexE[i]).reshape(
        TurnSize,) - i * T * 720
del Data

# combine all data (different turns) together
NewTime = np.sort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpIndex = np.argsort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpWave = np.reshape(TurnData, (TurnSize * TurnNum,))
del TurnTime, TurnData
NewWave = tmpWave[tmpIndex]
NewTime = NewTime * 50
del tmpIndex, tmpWave

xx = np.arange(0.1, T * 50 * 720, 0.1)
f = interp1d(NewTime, NewWave, kind="linear", bounds_error=False, fill_value=0)
tmp = f(xx)
tmp[-10000:] = tmp[-20000:-10000]
windowSize = 100
b = (1 / windowSize) * np.ones((windowSize,))
a = 1
LUTtmp = signal.filtfilt(b, a, tmp)
N = np.floor(len(LUTtmp) / 720).astype("int32")
LUT = LUTtmp[:N * 720].reshape((N, 720), order="F")
del tmp, LUTtmp, NewTime, NewWave, f, b

Data = BPM1[PeakIndex - 10:].copy()
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# build LUT matrix for the bunch #BunchIndex and turn #TurnNum
tmp1 = LUT[:, BunchIndex]
tmp2 = np.concatenate((tmp1, tmp1, tmp1))
LutMatrix = np.zeros((40, 20000)).astype("float64")
for i in range(20000):
    LutMatrix[:, i] = tmp2[np.arange(i + 1000, i + 20000 + 1000, 500)]
del tmp1, tmp2

# find the bunch phase using correclation method
DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                      T).astype("int32") + np.floor((BunchIndex *T)).astype("int32")
DataIndexE = DataIndexS + BunchSize

Data = Data.reshape(len(Data),)
BunchData = np.zeros((40, TurnNum)).astype("float64")
BunchPhase0 = np.zeros(TurnNum).astype("float64")
BunchMatrix = np.zeros((20000, 40)).astype("float64")
BunchPhaseFit = np.zeros(TurnNum)
for i in range(TurnNum):
    BunchData[:, i] = Data[np.arange(DataIndexS[i], DataIndexE[i])]
    BunchPhase0[i] = (DataIndexS[i] - i * T * 720 - BunchIndex * T) * 50
    DataMatrix = np.tile(BunchData[:, i], (20000, 1)).T
    tmp1 = np.mean(np.abs(LutMatrix - DataMatrix), axis=0)
    ind = np.argmin(tmp1)
    BunchPhaseFit[i] = ind
    print(i)

BunchPhaseFit = BunchPhaseFit * 0.1
BunchPhase = BunchPhase0 - BunchPhaseFit + 1000
x = np.arange(len(BunchPhase))
z1 = np.polyfit(x, BunchPhase, 1)
T = T + z1[0] / 720 / 50
del BunchData, BunchDataEnd, BunchDataFirst, BunchMatrix, BunchPhase0, BunchPhaseFit, DataIndexS, DataIndexE, LUT, \
    LutMatrix, tmp1, xx, z1


# build the final LUT of all bunches, using the final T value, pickup #1
TurnSize = np.floor(T * 720).astype("int32")
TurnNum = np.floor(len(Data) / 720 / T).astype("int32") - 1
# collect the all bunches data together using the new T value
DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                      T).astype("int32")
DataIndexE = DataIndexS + TurnSize
TurnData = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
TurnTime = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
for i in range(TurnNum):
    TurnData[:, i] = Data[DataIndexS[i]:DataIndexE[i]].reshape(TurnSize,)
    TurnTime[:, i] = np.arange(DataIndexS[i], DataIndexE[i]).reshape(
        TurnSize,) - i * T * 720
del Data
# combine all data (different turns) together
NewTime = np.sort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpIndex = np.argsort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpWave = np.reshape(TurnData, (TurnSize * TurnNum,))
del TurnTime, TurnData
NewWave = tmpWave[tmpIndex]
NewTime = NewTime * 50
del tmpIndex, tmpWave

xx = np.arange(0.1, T * 50 * 720, 0.1)
f = interp1d(NewTime, NewWave, kind="linear", bounds_error=False, fill_value=0)
tmp = f(xx)
tmp[-10000:] = tmp[-20000:-10000]
windowSize = 100
b = (1 / windowSize) * np.ones((windowSize,))
a = 1
LUTtmp = signal.filtfilt(b, a, tmp)
N = np.floor(len(LUTtmp) / 720).astype("int32")
LUT1 = LUTtmp[:N * 720].reshape((N, 720), order="F")
np.save("LUT1", LUT1)
del tmp, LUTtmp, NewTime, NewWave, f, b, LUT1

# build the final LUT of all bunches, using the final T value, pickup #3
Data = BPM3[PeakIndex - 10:].copy()
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
TurnSize = np.floor(T * 720).astype("int32")
TurnNum = np.floor(len(Data) / 720 / T).astype("int32") - 1
# collect the all bunches data together using the new T value
DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                      T).astype("int32")
DataIndexE = DataIndexS + TurnSize
TurnData = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
TurnTime = np.zeros((-DataIndexS[0] + DataIndexE[0], TurnNum))
for i in range(TurnNum):
    TurnData[:, i] = Data[DataIndexS[i]:DataIndexE[i]].reshape(TurnSize,)
    TurnTime[:, i] = np.arange(DataIndexS[i], DataIndexE[i]).reshape(
        TurnSize,) - i * T * 720
del Data
# combine all data (different turns) together
NewTime = np.sort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpIndex = np.argsort(TurnTime.reshape((TurnSize * TurnNum,)))
tmpWave = np.reshape(TurnData, (TurnSize * TurnNum,))
del TurnTime, TurnData
NewWave = tmpWave[tmpIndex]
NewTime = NewTime * 50
del tmpIndex, tmpWave
xx = np.arange(0.1, T * 50 * 720, 0.1)
f = interp1d(NewTime, NewWave, kind="linear", bounds_error=False, fill_value=0)
tmp = f(xx)
tmp[-10000:] = tmp[-20000:-10000]
windowSize = 100
b = (1 / windowSize) * np.ones((windowSize,))
a = 1
LUTtmp = signal.filtfilt(b, a, tmp)
N = np.floor(len(LUTtmp) / 720).astype("int32")
LUT2 = LUTtmp[:N * 720].reshape((N, 720), order="F")
np.save("LUT2", LUT2)
np.save("T", T)
del tmp, LUTtmp, NewTime, NewWave, f, b, LUT2
