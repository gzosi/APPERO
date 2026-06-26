import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset2'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 37]] 
        class DynamicROI:
            Camera1 = {
                0 : np.array([[751, 392], [615, 437], [424, 391], [358, 261], [530, 244], [741, 307]]),
                10 : np.array([ [741, 409], [590, 445], [403, 437], [300, 266], [471, 230], [711, 331]]),
                20 : np.array([[644, 427], [522, 473], [324, 480], [241, 314], [444, 280], [625, 338]]),
                30 : np.array([[535, 424], [350, 481], [277, 404], [267, 299], [387, 271], [519, 312]]),
                40 : np.array([[460, 420], [371, 492], [284, 433], [266, 324], [364, 302], [431, 336]]),
                }
            Camera2 = {
                0 : np.array([[532, 297], [352, 369], [183, 352], [109, 189], [306, 153], [490, 209]]),
                10 : np.array([[407, 312], [275, 353], [130, 338], [86, 184], [244, 192], [391, 235]]),
                20 : np.array([[338, 297], [188, 380], [88, 399], [6, 209], [180, 177], [318, 214]]),
                30 : np.array([[253, 340], [187, 375], [21, 385], [8, 234], [102, 179], [237, 239]]),
                40 : np.array([[217, 349], [126, 386], [40, 390], [11, 230], [94, 215], [156, 246]]),
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
                kernel = (9,9)
                sigma = 0          
                strength = 1.5 
        class Focus:
            class Entropy:
                disk_size = 8
            class HED:
                detect_resolution = 512
            class Weights:
                hed = 1
                entropy = 0.5
                diff = 1
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
            TargetPts = 3
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