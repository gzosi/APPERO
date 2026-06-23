import cv2 as cv
#%% Defining Config Packet
class Task1:
    class MetaData:
        InputExt = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        OutputName = 'Data.h5'
    class Settings:
        class Rotation:
            Camera1 = None
            Camera2 = cv.ROTATE_180
            # cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE 
        class Resync:
            class Database2:
                Dataset1 = 7
            class Database3: 
                class Dataset2:
                    Background = 24
                    Foreground = 25
            class Database4:
                Dataset1 = 30
                Dataset2 = 4
                Dataset3 = 21
                Dataset4 = 41
                Dataset5 = 23
                Dataset6 = 32
                Dataset7 = 59
                Dataset8 = 15
                Dataset9 = 12
                Dataset10 = 55
                Dataset11 = 34
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0