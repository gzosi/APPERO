keys = [8, 10, 12]
#####################################################################
#####################################################################
from Config import Config
from pathlib import Path
import h5py
import numpy as np
import cv2
import json # Puoi anche rimuovere questo import se non lo usi più altrove

# Funzione di callback per catturare i click del mouse
def select_point(event, x, y, flags, param):
    current_points = param['points']
    img_display = param['image']
    window_name = param['window_name']
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 6:
            current_points.append([x, y])
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            if len(current_points) > 1:
                cv2.line(img_display, tuple(current_points[-2]), tuple(current_points[-1]), (0, 255, 0), 2)
            if len(current_points) == 6:
                cv2.line(img_display, tuple(current_points[-1]), tuple(current_points[0]), (0, 255, 0), 2)
            cv2.imshow(window_name, img_display)

task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1
main_root = Path(Config.Paths.mainRooot)
srcRoot = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)

settings = task_conf.Settings
dynamic_roi_data = {}
window_name = 'Seleziona 6 punti per la ROI'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
quit_app = False

print("Istruzioni:")
print("- Clicca col mouse sinistro per selezionare 6 punti.")
print("- Dopo aver selezionato 6 punti, premi 'a' per Accettare o 's' per Scartare e riprovare.")
print("- Premi 'q' in qualsiasi momento di attesa per salvare ed uscire.\n")

with h5py.File(srcRoot, 'r') as f:
    cameras = list(f.keys())
    groups = {camera:
        f[camera][settings.Src.Database]['Dataset11']
        for camera in cameras}
    
    for camera in cameras:
        if quit_app: break
        dynamic_roi_data[camera] = {}
        for key in keys:
            if quit_app: break
            
            # Prova a leggere il frame, gestendo eventuali errori se la chiave non esiste
            try:
                raw = groups[camera][f"{key:05}"][:].astype(np.uint8)
            except KeyError:
                print(f"Frame {key:05} non trovato in {camera}. Salto...")
                continue

            while True:
                if len(raw.shape) == 2:
                    img_display = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                else:
                    img_display = raw.copy()
                    
                current_points = []
                mouse_params = {
                    'points': current_points,
                    'image': img_display,
                    'window_name': window_name
                }
                cv2.imshow(window_name, img_display)
                cv2.setMouseCallback(window_name, select_point, mouse_params)
                print(f"--- Lavorando su {camera}, Frame {key} ---")
                
                while len(current_points) < 6:
                    k = cv2.waitKey(10) & 0xFF
                    if k == ord('q'):
                        quit_app = True
                        break
                
                if quit_app: break
                
                print("6 punti selezionati. Premi 'a' (Accetta), 's' (Scarta) o 'q' (Esci).")
                action_taken = False
                while not action_taken:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('a'):
                        # Salva i punti nel dizionario
                        dynamic_roi_data[camera][key] = current_points
                        print("ROI Salvata!\n")
                        action_taken = True
                        break
                    elif k == ord('s'):
                        print("ROI Scartata. Riprova a selezionare i punti.\n")
                        action_taken = True
                        break 
                    elif k == ord('q'):
                        quit_app = True
                        action_taken = True
                        break
                
                if k == ord('a') or quit_app:
                    break

cv2.destroyAllWindows()

# ---------------------------------------------------------------------
# NUOVA SEZIONE DI SALVATAGGIO: Scrittura in formato Python (.py)
# ---------------------------------------------------------------------
output_file = 'dynamic_roi.py'

with open(output_file, 'w') as f:
    # Intestazione del file
    f.write("import numpy as np\n\n")
    
    # Iteriamo su ogni camera
    for camera, frames in dynamic_roi_data.items():
        # Se la camera non ha frame salvati, saltiamo
        if not frames:
            continue
            
        f.write(f"{camera} = {{\n")
        
        # Iteriamo sui frame (chiavi) di quella camera
        for key, points in frames.items():
            if len(points) == 6:
                # Dividiamo i punti in due blocchi da 3 per replicare la tua formattazione
                riga1 = ", ".join([f"[{p[0]}, {p[1]}]" for p in points[:3]])
                riga2 = ", ".join([f"[{p[0]}, {p[1]}]" for p in points[3:]])
                
                f.write(f"    {key} : np.array([\n")
                f.write(f"        {riga1}, \n")
                f.write(f"        {riga2}]),\n")
            else:
                # Fallback nel caso in cui i punti non siano esattamente 6 (non dovrebbe accadere per la tua logica)
                f.write(f"    {key} : np.array({points}),\n")
                
        f.write("}\n\n")

print(f"\nOperazione completata! I dati sono stati salvati in '{output_file}' in formato codice Python.")