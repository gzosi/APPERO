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
            Dataset = 'Dataset4'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 45]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[798, 164], [616, 318], [455, 429], [392, 334], [534, 161], [799, 30]]),
                10 : np.array([[855, 218], [679, 362], [427, 476], [403, 350], [576, 179], [832, 68]]),
                20 : np.array([[846, 356], [637, 484], [396, 544], [353, 354], [639, 194], [842, 135]]),
                30 : np.array([[837, 473], [609, 560], [337, 579], [323, 443], [592, 306], [811, 234]]),
                40 : np.array([[828, 534], [548, 617], [355, 609], [336, 505], [559, 387], [791, 282]]),
                50 : np.array([[771, 641], [556, 677], [330, 631], [332, 532], [574, 455], [736, 387]]),
            }
            Camera2 = {
                0 : np.array([[800, 249], [642, 399], [445, 503], [428, 383], [586, 245], [805, 129]]),
                10 : np.array([[887, 292], [695, 473], [471, 577], [441, 439], [646, 261], [872, 161]]),
                20 : np.array([[880, 455], [699, 577], [442, 624], [440, 479], [654, 348], [867, 250]]),
                30 : np.array([[875, 559], [698, 655], [433, 656], [423, 517], [660, 373], [870, 301]]),
                40 : np.array([[876, 673], [684, 690], [422, 671], [417, 593], [640, 457], [874, 393]]),
                50 : np.array([[883, 686], [678, 691], [440, 690], [471, 624], [682, 519], [875, 482]]),
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