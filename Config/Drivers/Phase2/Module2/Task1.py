#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        class ICP:
            Activation = True
            Threshold = 10.0 
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0