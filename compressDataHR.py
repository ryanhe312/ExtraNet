import numpy as np
import cv2
import os
import sys
import glob
from PIL import Image
from multiprocessing import Process
from torchvision import transforms
import torch
# basePath0 = "D:/training_set_v2/WithoutDemodulate/MedievalOrigin/"
# basePath1 = "I:/"

threadNum = 4

# dirList = ["G:/Unreal Projects/SunSet/Saved/VideoCaptures/"] ## Must end with /
# dirList = ["G:/Unreal Projects/Blueprints/Blueprints/Saved/VideoCaptures/"] ## Must end with /
dirList = ["G:/Unreal Projects/Lewis/Saved/VideoCaptures_Test/HR/"]

# ScenePrefix = "ThirdPersonExampleMap"
# ScenePrefix = "BlueprintOffice"
ScenePrefix = "DemoMap2"
GtPrefix = "FinalImage"
# DepthPrefix = "SceneDepth"
# MVPrefix ="MotionVector"




def MergeRange(start, end, inPath, outPath):
    for idx in range(start, end):
        newIdx = str(idx).zfill(4)
        
        img = cv2.imread(inPath+"/"+ScenePrefix+GtPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)[:,:,:3]
        # img = cv2.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)

        img = torch.nn.functional.interpolate(torch.tensor(img).permute(2,0,1).unsqueeze(0),scale_factor=0.5,mode='bilinear',align_corners=True)

        # print(img.shape)

        img = img.squeeze().permute(1,2,0).numpy().astype(np.float16)
        print('outputing',outPath+'compressedHR.{}.npz'.format(newIdx))
        np.savez_compressed(outPath+'compressedHR.{}.npz'.format(newIdx), i = img)
        # np.save(outPath+'compressed.{}'.format(newIdx), res)
def GetCompressStartEnd(path):
    start = 99999
    end = 0

    for filePath in glob.glob(path + "/*"):
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    return start, end
def GetStartEnd(path):
    start = 99999
    end = 0

    for filePath in glob.glob(path + "/"+ScenePrefix+GtPrefix+"*"):
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    return start, end
def MergeFile(inPath, outPath):
    # get index range
    start, end = GetStartEnd(inPath)

    if not os.path.exists(outPath):
        os.mkdir(outPath)
    # combine
    processList = list()
    part = (end - start + 1) // threadNum + 1
    for t in range(threadNum):
        processList.append(Process(target=MergeRange, args=(start+t*part, min(start+t*part+part, end+1),inPath, outPath,)))

    for sub_process in processList:
        sub_process.start()
    for sub_process in processList:
        sub_process.join()

def Compress32To16(start, end, path):
    for idx in range(start, end):
        fileName = path+"/compressed.{}.npy".format(str(idx).zfill(4))
        try:
            total = np.load(fileName)
        except:
            continue
        total = total.astype(np.float16)
        np.save(fileName, total)

def CompressRange(di):
    start, end = GetCompressStartEnd(di)
    print(start)
    print(end)
    # combine
    processList = list()
    part = (end - start + 1) // threadNum + 1
    for t in range(threadNum):
        processList.append(Process(target=Compress32To16, args=(start+t*part, min(start+t*part+part, end+1),di,)))

    for sub_process in processList:
        sub_process.start()
    for sub_process in processList:
        sub_process.join()

if __name__ == "__main__":
    for di in dirList:
        MergeFile(di, di)
    # for di in dirList:
    #     CompressRange(di)

    # Compress32To16(r"D:\NoAACompressed\Medieval_0\compressed.0011.npy")
    # ReadData(r"D:\NoAACompressed\Medieval_0\compressed.0011.npy")
    # ReadData(r"E:\taa\compressed.0180.npy")