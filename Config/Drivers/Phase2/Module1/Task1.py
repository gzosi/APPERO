#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Src:
            Database = 'Database2'
            Dataset = 'Dataset1'
        class Ref:
            Database = 'Database1'
            Dataset = 'Dataset1'
            Record = 'Record1'
        class Calib:
            Dataset = 'Dataset1'
            Model = 'Model27'
        class Syncrony:
            Bounds = [(0, 255)]
        EpiLimit = 50
        ConfLimit = 0
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0