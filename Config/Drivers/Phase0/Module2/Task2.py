#%% Defining Config Packet
class Task2:
    class MetaData:
        ShapeExt = 'DataShape.json'
        OriginExt = 'DataOrigin.json'
        Indent = 4
    class Settings:
        IncludeOrigin = True 
        FullSensorShape = (1280, 800)
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0