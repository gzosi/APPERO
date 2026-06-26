import numpy as np
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset1'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 45]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[444, 479], [253, 344], [142, 192], [162, 97], [355, 208], [479, 379]]),
                10 : np.array([[417, 464], [256, 371], [119, 242], [130, 172], [316, 250], [448, 377]]),
                20 : np.array([[409, 505], [224, 415], [117, 332], [124, 231], [289, 289], [426, 417]]),
                30 : np.array([[431, 510], [253, 482], [129, 398], [137, 319], [298, 351], [454, 437]]),
                40 : np.array([[463, 563], [293, 582], [134, 518], [142, 389], [301, 473], [441, 495]]),
            }
            Camera2 = {
                0 : np.array([[463, 609], [323, 490], [235, 349], [269, 296], [426, 405], [511, 541]]),
                10 : np.array([[428, 569], [291, 508], [191, 392], [195, 343], [359, 405], [456, 495]]),
                20 : np.array([[431, 601], [311, 582], [183, 500], [182, 393], [364, 471], [460, 519]]),
                30 : np.array([[475, 652], [337, 645], [208, 579], [193, 470], [358, 518], [497, 582]]),
                40 : np.array([[456, 674], [329, 680], [200, 634], [195, 534], [321, 565], [445, 633]]),
            }
        class Enhancement:
            class Bilateral:
                d = 1                 
                sigmaColor = 50       
                sigmaSpace = 50       
            class CLAHE:
                clipLimit = 15.0       
                tileGridSize = (8, 8)
            class UnsharpMask:
                kernel = (3,3)
                sigma = 0          
                strength = 2.5 
        class Focus:
            class Entropy:
                disk_size = 8
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1
                entropy = 0.75
                diff = 0.5
            class Morph:
                class Kernels:
                    Open = (5,5)
                    Dilate = (9,9)
                    Erode = (9,9)
                class Its:
                    Dilate = 5
                    Erode = 7
                AreaMin = 150
            class EmptyCheck:   
                TopPercent = 0.5
                MeanThresh = 50
        class SmartPrompt:
            MinArea = 150
            TargetPts = 5
            NegativeRing = 15
            PercentileLevels = [30,60, 90]
            IouThreshold = 0.75
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
        class Group:
            AreaMin = 150             
            Similarity = 0.25     
        class Collapse:
                PercentileThresh = 10.0    
                MinMaxPercentage = 0.1
        class Cloud:
            BlurKernel = [9, 9]     
            Relaxation = 0.85     
            DilateKernel = [9, 9]  
            DilateIter = 1       
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0  