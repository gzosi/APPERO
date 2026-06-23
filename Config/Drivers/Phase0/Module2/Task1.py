#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        InputExt = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        OutputName = 'Data.h5'
    class Settings:
        class Rotation:
            Camera1 = None
            Camera2 = None
            # cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE 
        class Resync:
            class Database2:
                Dataset1 = 75
            class Database3: 
                class Dataset1:
                    Background = 28
                    Foreground = 30
            class Database4:
                Dataset1 = 30
                Dataset2 = 30
                Dataset3 = 30
                Dataset4 = 30
                Dataset5 = 30
                Dataset6 = 30
                Dataset7 = 30
                Dataset8 = 30
                Dataset9 = 30
                Dataset10 = 30
                Dataset11 = 30
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0