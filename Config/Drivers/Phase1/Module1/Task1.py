#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        Database = 'Database1'
    class Parameters:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (11, 11) 
        zeroZone = (-1, -1)
        class Flags:
            typeA = cv.CALIB_CB_FILTER_QUADS | cv.CALIB_CB_ACCURACY
            typeB = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY | cv.CALIB_CB_LARGER
                # cv.CALIB_CB_ADAPTIVE_THRESH |
                # cv.CALIB_CB_FAST_CHECK | 
                # cv.CALIB_CB_NORMALIZE_IMAGE |
                # cv.CALIB_CB_FILTER_QUADS |
                # cv.CALIB_CB_EXHAUSTIVE |
                # cv.CALIB_CB_ACCURACY |
                # cv.CALIB_CB_CLUSTERING
        class Subpix:
            typeA = True
            typeB = False
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0