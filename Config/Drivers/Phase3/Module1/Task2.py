class Task2:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset4'
            Foreground = 'Foreground'
        class Ref:
            Database = 'Database1'
            Dataset = 'Dataset4'
            Record = 'Record1'
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        EpiLimit = 100000
        ConfLimit = 0
        SimLimit = 0.1
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0