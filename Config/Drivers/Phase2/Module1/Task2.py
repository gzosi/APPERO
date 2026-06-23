#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Src:
            Database = 'Database2'
            Dataset = 'Dataset1'
        class Ref:
            Database = 'Database1'
            Dataset = 'Dataset4'
            Record = 'Record1'
    class Tracking:
        LKparam = dict(
            winSize=(21, 21), # Finestra abbastanza grande per catturare il marker
            maxLevel=3,       # Piramide a 3 livelli per gestire piccoli blur
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        distLimit = 5
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0