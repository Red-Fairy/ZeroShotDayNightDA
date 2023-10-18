import numpy as np
import os
import pickle

def create_pkl_gp(dataPath,ref):
    refId = ref[0]+ref[4]
    dbImgList = sorted(os.listdir(os.path.join(dataPath,ref)))
    qImgList = sorted(os.listdir(os.path.join(dataPath,"night_right")))
    dbImgList = [os.path.join(ref,n) for n in dbImgList]
    qImgList = [os.path.join('night_right',n) for n in qImgList]
    numQ = len(qImgList)
    numDb = len(dbImgList)
    qMatchList = [np.arange( max(0,n-5), min((n+6),numDb) ) for n in range(numQ)]
    gnd = [{'ok':qMatchList[i]} for i in range(numQ)]
    pklDict = {'imlist':dbImgList,'qimlist':qImgList,'gnd':gnd}

    savePath = "./data/test/gp_{}_nr/".format(refId)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    gnd_fname = "./data/test/gp_{}_nr/gnd_gp_{}_nr.pkl".format(refId,refId)
    with open(gnd_fname, 'wb') as f:
        cfg = pickle.dump(pklDict,f)

if __name__ == "__main__":
    dataPath = "./data/test/GardensPointWalking/"
    create_pkl_gp(dataPath,'day_left')
    create_pkl_gp(dataPath,'day_right')
