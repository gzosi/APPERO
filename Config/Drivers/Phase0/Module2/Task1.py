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
        class Resync:
            class Database2:
                Dataset1 = 67
            class Database3: 
                class Dataset4:
                    Background = 16
                    Foreground = 22
            class Database4:
                Dataset1 = 16
                Dataset2 = 32
                Dataset3 = 36
                Dataset4 = 31
                Dataset5 = 1
                Dataset6 = 36
                Dataset7 = 4
                Dataset8 = 37
                Dataset9 = 56
                Dataset10 = 15
                Dataset11 = 0
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0