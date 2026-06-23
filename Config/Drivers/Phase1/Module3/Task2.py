#%% Importing Libreries
import cv2 as cv
#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        Database = 'Database1'
    class Parameters:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        flag = cv.CALIB_FIX_INTRINSIC | cv.CALIB_FIX_PRINCIPAL_POINT | cv.CALIB_FIX_FOCAL_LENGTH
        class Models:
            class Model1:
                name = 'none'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model2:
                name = 'radialBase'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model3:
                name = 'radialExtra'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model4:
                name = 'tangential'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model5:
                name = 'thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model6:
                name = 'tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model7:
                name = 'radialBase_radialExtra'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model8:
                name = 'radialBase_tangential'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model9:
                name = 'radialBase_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model10:
                name = 'radialBase_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model11:
                name = 'radialExtra_tangential'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model12:
                name = 'radialExtra_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model13:
                name = 'radialExtra_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model14:
                name = 'tangential_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model15:
                name = 'tangential_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model16:
                name = 'thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST)
            class Model17:
                name = 'radialBase_radialExtra_tangential'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_S1_S2_S3_S4 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model18:
                name = 'radialBase_radialExtra_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model19:
                name = 'radialBase_radialExtra_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_ZERO_TANGENT_DIST +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model20:
                name = 'radialBase_tangential_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model21:
                name = 'radialBase_tangential_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model22:
                name = 'radialBase_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6 +
                    cv.CALIB_ZERO_TANGENT_DIST)
            class Model23:
                name = 'radialExtra_tangential_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model24:
                name = 'radialExtra_tangential_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model25:
                name = 'radialExtra_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_ZERO_TANGENT_DIST)
            class Model26:
                name = 'tangential_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6)
            class Model27:
                name = 'radialBase_radialExtra_tangential_thinPrism'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_TAUX_TAUY)
            class Model28:
                name = 'radialBase_radialExtra_tangential_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_S1_S2_S3_S4)
            class Model29:
                name = 'radialBase_radialExtra_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_ZERO_TANGENT_DIST)
            class Model30:
                name = 'radialBase_tangential_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6)
            class Model31:
                name = 'radialExtra_tangential_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL +
                    cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3)
            class Model32:
                name = 'radialBase_radialExtra_tangential_thinPrism_tilted'
                value = (
                    cv.CALIB_RATIONAL_MODEL + cv.CALIB_THIN_PRISM_MODEL + cv.CALIB_TILTED_MODEL)
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0