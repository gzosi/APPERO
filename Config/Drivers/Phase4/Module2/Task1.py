#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        class Carver:
            occupancyLimit = 2
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0