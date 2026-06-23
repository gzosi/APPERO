#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        sizes = dict(
            Dataset1 = dict(
                Record1 = 50,
                Record2 = 50),
            Dataset2 = dict(
                Record1 = 50,
                Record2 = 50),     
            Dataset3 = dict(
                Record1 = 50,
                Record2 = 50), 
        ) # seleziona la quantità di immagini da estrarre dalle diverse registrazioni
        types = dict(
            Dataset1 = dict(
                Record1 = ['typeA'],
                Record2 = ['typeA']),
            Dataset2 = dict(
                Record1 = ['typeA'],
                Record2 = ['typeA']),     
            Dataset3 = dict(
                Record1 = ['typeA'],
                Record2 = ['typeA']), 
        ) # seleziona la tipologia di punti da utilizzare ['typeA', 'typeB'] 
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0