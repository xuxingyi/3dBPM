import scipy.io as scio
import numpy as np
BunchAmp1 = np.load("BunchAmp1.npy")
BunchPhase1 = np.load("BunchPhase1.npy")
BunchAmp3 = np.load("BunchAmp3.npy")
BunchPhase3 = np.load("BunchPhase3.npy")
scio.savemat("20200220_213mA_AC_inject_6_Phase.mat",{'BunchPhase1':BunchPhase1,'BunchPhase3':BunchPhase3})
scio.savemat("20200220_213mA_AC_inject_6_Amp.mat",{'BunchAmp1':BunchAmp1,'BunchAmp3':BunchAmp3})