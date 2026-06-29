import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Src:
            Database = 'Database4'
        class Ref:
            Database = 'Database3'
            Dataset = 'Dataset4'
            Location = 'Background'
        Bounds = [[30, 40]] 
        class DynamicROI:
            Camera1 = {
                30 : np.array([[808, 494], [678, 569], [456, 569], [413, 465], [523, 350], [681, 286]]),
                35 : np.array([[804, 550], [641, 604], [454, 593], [412, 466], [562, 356], [718, 334]]),
                40 : np.array([[797, 596], [615, 630], [442, 595], [431, 491], [568, 402], [729, 382]]),
            }
            Camera2 = {
                30 : np.array([[885, 593], [722, 645], [508, 651], [476, 494], [667, 379], [860, 365]]),
                35 : np.array([[873, 637], [699, 682], [525, 683], [513, 552], [666, 452], [853, 423]]),
                40 : np.array([[886, 683], [706, 691], [477, 640], [464, 527], [667, 425], [865, 413]]),
            }
        class Zscore:
            zTh = 3.0
            epsilon = 1e-5
            class Morph:
                openKernel = (3,3)
                closeKernel = (5,5)
        class SmartPrompt:
            MinArea = 10             
            TargetPts = 4           
            NumNegativePts = 8      
            NegativeRingBase = 8   
            BboxPaddingBase = 8     
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
        class ContainmentZone:
            Kernel = (25,25)      
            minArea = 10        
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0