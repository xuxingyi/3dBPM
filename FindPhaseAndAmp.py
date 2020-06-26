import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import signal

# Get the right RF frequency information from BPM signal
filename = "D:/study/数据处理脚本及信号处理流程演示/BunchTrain97600uAatt03.mat"
load_data = sio.loadmat(filename)
BPM1 = np.array(load_data["BPM1"], dtype="int32")
BPM3 = np.array(load_data["BPM3"], dtype="int32")
T = np.load("T.npy")
LUT1 = np.load("LUT1.npy")
LUT2 = np.load("LUT2.npy")
PhaseBalance = np.load("PhaseBalance.npy")

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

BaselineIndex = np.arange(PeakIndex - 3688, PeakIndex - 688)
Filling = np.arange(500)

# define the basic number
HarmonicNum = 720
# initial bucket size value, sampling rate 20GHz, 40*50ps = 2ns
BunchSize = 40
# define which bunch will be processed
BunchIndexScan = Filling

# copy raw data, processing pickup #1
Data = np.array(load_data["BPM1"], dtype="int32")[PeakIndex - 10:].copy()
LUT = LUT1
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# for loop to process all defined bunch
N = len(BunchIndexScan)
LUTlength = 2000
LUTstart = 18000
TurnNum = np.floor(len(Data) / 720 / T).astype("int32") - 1
LutMatrix = np.zeros((40, LUTlength)).astype("float64")
BunchData = np.zeros((40, TurnNum)).astype("float64")
BunchPhase0 = np.zeros(TurnNum).astype("float64")
DataMatrix = np.zeros((40, LUTlength)).astype("float64")
BunchPhaseFit = np.zeros(TurnNum).astype("float64")
BunchDataFit = np.zeros((40, TurnNum)).astype("float64")
BunchPhase = np.zeros((TurnNum, N)).astype("float64")
BunchAmp = np.zeros((N, TurnNum)).astype("float64")
TurnNum = np.floor(len(Data) / 720 / T).astype('int32') - 1
for j in range(N):
    print(j)
    BunchIndex = BunchIndexScan[j]
    # define the data index for the bunch in the first turn
    DataIndexStart = BunchIndex * BunchSize
    # build LUT matrix for the bunch  # BunchIndex
    tmp1 = LUT[:, BunchIndex]
    tmp2 = np.concatenate((tmp1, tmp1, tmp1))
    for i in range(LUTlength):
        LutMatrix[:, i] = tmp2[np.arange(
            LUTstart + i + 1000, LUTstart + i + 20000 + 1000, 500)]
    # collect the bunch data using the final T value
    DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                          T).astype("int32") + DataIndexStart
    DataIndexE = DataIndexS + BunchSize
    for i in range(TurnNum):
        BunchData[:, i] = Data[np.arange(
            DataIndexS[i], DataIndexE[i])].reshape((BunchSize,))
        BunchPhase0[i] = (DataIndexS[i] - i * T * 720 - BunchIndex * T) * 50
        DataMatrix = np.tile(BunchData[:, i], (LUTlength, 1)).T
        tmp3 = np.mean(LutMatrix * DataMatrix, axis=0)
        ind = np.argmax(tmp3)
        #print("ind=", ind)
        BunchDataFit[:, i] = LutMatrix[:, ind]
        BunchPhaseFit[i] = ind * 0.1
        BunchPhase[i, j] = BunchPhase0[i] - BunchPhaseFit[i] + \
            (T - 40) * 50 * BunchIndex - PhaseBalance[BunchIndex]
        z1 = np.polyfit(BunchDataFit[:, i], BunchData[:, i], 1)
        BunchAmp[j, i] = z1[0]

BunchPhase1 = np.copy(BunchPhase)
BunchAmp1 = np.copy(BunchAmp)

# copy raw data, processing pickup #3
Data = np.array(load_data["BPM3"], dtype="int32")[PeakIndex - 10:].copy()
LUT = LUT2
# remove DC offset
Baseline = np.mean(Data[BaselineIndex], dtype="float64")
Data = Data - Baseline
# for loop to process all defined bunch
N = len(BunchIndexScan)
LUTlength = 2000
LUTstart = 18000
for j in range(N):
    BunchIndex = BunchIndexScan[j]
    # define the data index for the bunch in the first turn
    DataIndexStart = BunchIndex * BunchSize
    # build LUT matrix for the bunch  # BunchIndex
    tmp1 = LUT[:, BunchIndex]
    tmp2 = np.concatenate((tmp1, tmp1, tmp1))
    for i in range(LUTlength):
        LutMatrix[:, i] = tmp2[np.arange(
            LUTstart + i + 1000, LUTstart + i + 20000 + 1000, 500)]
    # collect the bunch data using the final T value
    DataIndexS = np.floor(np.arange(TurnNum) * 720 *
                          T).astype("int32") + DataIndexStart
    DataIndexE = DataIndexS + BunchSize
    for i in range(TurnNum):
        BunchData[:, i] = Data[np.arange(
            DataIndexS[i], DataIndexE[i])].reshape((BunchSize,))
        BunchPhase0[i] = (DataIndexS[i] - i * T * 720 - BunchIndex * T) * 50
        DataMatrix = np.tile(BunchData[:, i], (LUTlength, 1)).T
        tmp3 = np.mean(LutMatrix * DataMatrix, axis=0)
        ind = np.argmax(tmp3)
        # print("ind=", ind)
        BunchDataFit[:, i] = LutMatrix[:, ind]
        BunchPhaseFit[i] = ind * 0.1
        BunchPhase[i, j] = BunchPhase0[i] - BunchPhaseFit[i] + \
            (T - 40) * 50 * BunchIndex - PhaseBalance[BunchIndex]
        z1 = np.polyfit(BunchDataFit[:, i], BunchData[:, i], 1)
        BunchAmp[j, i] = z1[0]

BunchPhase3 = np.copy(BunchPhase)
BunchAmp3 = np.copy(BunchAmp)

np.save("BunchPhase1", BunchPhase1)
np.save("BunchPhase3", BunchPhase3)
np.save("BunchAmp1", BunchAmp1)
np.save("BunchAmp3", BunchAmp3)
