#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pk'
    class Settings:
        sizes = dict(
            Dataset1 = dict(
                Record1 = 50,
                Record2 = 50),
        ) # seleziona la quantità di immagini da estrarre dalle diverse registrazioni
        types = dict(
            Dataset1 = dict(
                Record1 = ['typeA', 'typeB'],
                Record2 = ['typeA', 'typeB']),
        ) # seleziona la tipologia di punti da utilizzare ['typeA', 'typeB'] 
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0