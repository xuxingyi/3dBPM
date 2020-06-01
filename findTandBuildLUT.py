import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Get the right RF frequency information from BPM signal
filename = "C:/Users/74506/Desktop/数据处理脚本及信号处理流程演示/BunchTrain97600uAatt03.mat"
load_data = sio.loadmat(filename)
BPM1 = np.array(load_data["BPM1"],dtype="int32")
BPM3 = np.array(load_data["BPM3"],dtype="int32")

# define the peak index of the first bunch
flagForBaseline = False
for i in range(np.size(BPM1)//100):
    if(flagForBaseline==False):
        if((max(BPM1[i*100:(i+1)*100])-min(BPM1[i*100:(i+1)*100]))<4000):
            flagForBaseline = True
    elif((max(BPM1[i*100:(i+1)*100])-min(BPM1[i*100:(i+1)*100]))>4000):
        for k in range(270):
            if((BPM1[(i-1)*100+k+10]==min(BPM1[(i-1)*100+k:(i-1)*100+k+20])) and (max(BPM1[(i-1)*100+k:(i-1)*100+k+20])-min(BPM1[(i-1)*100+k:(i-1)*100+k+20]))>4000):
                PeakIndex = (i-1)*100+k+10
                break
        break

BaselineIndex = np.arange(PeakIndex-3688,PeakIndex-688)
#Filling = np.arange(499)


# find the right T using pickup #1
# process BPM1 signal
# copy raw data, include the previous 2 small bunches

Data = BPM1[PeakIndex-10:].copy()
# remove DC offset
Baseline = np.mean(Data[BaselineIndex],dtype="float64")
Data = Data - Baseline
del BPM1,BPM3

# define the basic number
HarmonicNum = 720
# initial bucket size value, sampling rate 20GHz, 40*50ps = 2ns
T = 40
BunchSize = 40
# define which bunch will be processed
BunchIndex = 0
# define the data index for the first bunch and the dirst turn
DataIndexStart = BunchIndex * BunchSize

# calculate T value by counting the bunch numbers
turnNumT = np.size(Data)//28000000
turnNum1 = np.array([2,5,10,20,100,200,500])
turnNum2 = np.arange(1,turnNumT+1)
ScanTurnNum = np.hstack((turnNum1,turnNum2*1000))
N = len(ScanTurnNum)
BunchDataFirst = np.zeros((BunchSize),dtype="float64")
BunchDataEnd = np.zeros((BunchSize),dtype="float64")
for j in range(N):
    TurnNum = ScanTurnNum[j]
    DataIndexS = np.floor(np.arange(TurnNum)*720*T).astype("int32") + DataIndexStart
    DataIndexE = DataIndexS + BunchSize
    # collect specified bunch data together
    BunchDataFirst = Data[DataIndexS[0]:DataIndexE[0]]
    BunchDataEnd = Data[DataIndexS[TurnNum-1]:DataIndexE[TurnNum-1]]
    # find the peak index for each turn of this bunch
    IndEnd = np.argmin(BunchDataEnd)
    IndFirst = np.argmin(BunchDataFirst)
    Peak1Index = IndFirst + DataIndexS[0] - 1
    PeakEndIndex = IndEnd + DataIndexS[-1] -1
    # calculate the T using peak index
    T = (PeakEndIndex - Peak1Index)/720/(TurnNum-1)

# build LUT of all bunches
TurnSize = np.floor(T*720).astype("int32")
TurnNum = np.floor(len(Data)/720/T).astype("int32")-1
# collect the all bunches data together using the new T value
DataIndexS = np.floor(np.arange(TurnNum) * 720 * T).astype("int32") + DataIndexStart
DataIndexE = DataIndexS + TurnSize
TurnData = np.zeros((-DataIndexS[0] + DataIndexE[0],TurnNum))
TurnTime = np.zeros((-DataIndexS[0] + DataIndexE[0],TurnNum))
for i in range(TurnNum):
    TurnData[:,i] = Data[DataIndexS[i]:DataIndexE[i]].reshape(TurnSize,)
    TurnTime[:,i] = np.arange(DataIndexS[i],DataIndexE[i]).reshape(TurnSize,) - i*T*720
del Data

# combine all data (different turns) together
NewTime = np.sort(TurnTime.reshape((TurnSize*TurnNum,)))
tmpIndex = np.argsort(TurnTime.reshape((TurnSize*TurnNum,)))
tmpWave = np.reshape(TurnData,(TurnSize*TurnNum,))
del TurnTime,TurnData
NewWave = tmpWave[tmpIndex]
NewTime = NewTime * 50
del tmpIndex,tmpWave,TurnTime

xx = np.arange(0.1,T*50*720,0.1)





