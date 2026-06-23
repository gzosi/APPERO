class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        Scroll = 20
        class ICP:
            usePointToPlane = True
            maxDistance = 5
            maxIterations = 10
        class LocalFilter:
            class Size:
                voxelSize = 0.5
            class Radius:
                nbPoints = 25
                radius = 5.0
            class Stats:
                nbNeighbors = 100
                stdRatio = 0.25
    class General:
        Activation = True
        Maker = True
        Destroyer = False