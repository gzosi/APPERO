import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import pickle
import json
from Config import Config

# =============================================================================
# CONFIGURAZIONE PERCORSI (Da adattare ai tuoi path reali)
# =============================================================================
# Cartella dove Task2.py ha salvato i file .pkl (dstRoot nel tuo script)
task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2
main_root = Path(Config.Paths.mainRooot)

RESULTS_DIR = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase3.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
    task_conf.__name__)

# Percorso al file HDF5 contenente le immagini sorgenti (srcRoot nel tuo script)
HDF5_PATH = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)

# Percorsi ai JSON per le forme e le origini (necessari per il padding delle immagini originali)
SHAPES_JSON_PATH = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot /
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)
ORIGINS_JSON_PATH = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot /
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__ / 
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt)


# Nomi delle videocamere usati come colonne nel DataFrame
CAMERAS = ['Camera1', 'Camera2'] # Sostituisci con i nomi reali (es. 'sx', 'dx')
# Parametri Visivi
ALPHA = 0.2          # Trasparenza delle maschere
DELAY_MS = 500       # Tempo di persistenza a schermo in millisecondi (0.5 sec)
MAX_SCREEN_W = 1600  # Larghezza massima della finestra sul monitor
MAX_SCREEN_H = 900   # Altezza massima della finestra sul monitor
# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================
def padOrigin(img, shape, origin):
    """Stessa funzione del tuo Task2.py per allineare l'immagine originale."""
    result = np.zeros((int(shape[0]), int(shape[1])), dtype=img.dtype)
    y_start, x_start = int(origin[1]), int(origin[0])
    y_end = y_start + img.shape[0]
    x_end = x_start + img.shape[1]
    result[y_start:y_end, x_start:x_end] = img
    return result
def apply_overlay(img_bgr, mask, color, alpha=0.3):
    """Applica una maschera colorata semi-trasparente sopra un'immagine."""
    if mask is None or np.count_nonzero(mask) == 0:
        return img_bgr
    
    # Crea un layer colorato
    color_layer = np.zeros_like(img_bgr, dtype=np.uint8)
    color_layer[mask > 0] = color
    
    # Crea una maschera booleana a 3 canali per fondere solo dove c'è la mask
    mask_3c = np.stack([mask > 0]*3, axis=-1)
    
    # Applica l'addWeighted solo nelle zone della maschera per non scurire il resto
    blended = np.where(mask_3c, cv2.addWeighted(img_bgr, 1 - alpha, color_layer, alpha, 0), img_bgr)
    return blended
def draw_points(img_bgr, pts, color, radius=3):
    """Disegna i punti (keypoints) sull'immagine."""
    if len(pts) == 0:
        return img_bgr
    
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_bgr, (x, y), radius, color, -1) # -1 riempie il cerchio
    return img_bgr

def resize_to_fit(img, max_w, max_h):
    """Ridimensiona l'immagine mantenendo l'aspect ratio per farla stare nello schermo."""
    h, w = img.shape[:2]
    if w > max_w or h > max_h:
        scaling_factor = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# =============================================================================
# MAIN SCOPE (VISUALIZZATORE)
# =============================================================================
def run_scope():
    print("Avvio dello Scope... Premi 'q' per uscire, 'p' per mettere in pausa/riprendere.")
    
    # Colori per i layer (BGR in OpenCV)
    COLOR_CAVITY = (0, 0, 255)   # Rosso
    COLOR_CLOUD = (255, 0, 0)    # Blu
    COLOR_PTS_CAVITY = (0, 255, 255) # Giallo per i punti cavità
    COLOR_PTS_CLOUD = (0, 255, 0)    # Verde per i punti cloud

    # 1. Caricamento dinamico delle origini per evitare l'IndexError
    try:
        with open(ORIGINS_JSON_PATH, 'r') as f:
            origins_dict = json.load(f)
        # Usiamo le stesse chiavi usate sotto per il file HDF5
        origin1 = origins_dict[CAMERAS[0]]['Database3'][task_conf.Settings.Src.Dataset]['Foreground']
        origin2 = origins_dict[CAMERAS[1]]['Database3'][task_conf.Settings.Src.Dataset]['Foreground']
    except Exception as e:
        print(f"Attenzione: Impossibile leggere origins.json ({e}). Uso origine (0,0).")
        origin1 = [0, 0]
        origin2 = [0, 0]

    # 2. Apre il file HDF5 in sola lettura
    try:
        f_h5 = h5py.File(HDF5_PATH, 'r')
        group1 = f_h5[CAMERAS[0]]['Database3'][task_conf.Settings.Src.Dataset]['Foreground'] 
        group2 = f_h5[CAMERAS[1]]['Database3'][task_conf.Settings.Src.Dataset]['Foreground']
    except Exception as e:
        print(f"Attenzione: Impossibile caricare l'HDF5 ({e}). Verranno usati sfondi neri.")
        f_h5, group1, group2 = None, None, None

    # 3. Cerca tutti i file generati (Supporta estensione .pkl)
    pkl_files = sorted(RESULTS_DIR.glob("*.pkl"))
    if not pkl_files:
        print("Nessun file .pkl trovato nella cartella specificata.")
        return

    paused = False

    for pkl_file in pkl_files:
        key = ''.join(filter(str.isdigit, pkl_file.stem))
        df = pd.read_pickle(pkl_file)
        
        # Carica le immagini raw dal DB HDF5 solo una volta per DataFrame
        img1_raw, img2_raw = None, None
        if f_h5 and key in group1 and key in group2:
            img1_raw = group1[key][:].astype(np.uint8)
            img2_raw = group2[key][:].astype(np.uint8)

        for idx, row in df.iterrows():
            # Formato salvato nel Task2.py: [padCavity, padCloud, cavityMatched, cloudMatched]
            cav_m1, cld_m1, cav_p1, cld_p1 = row[CAMERAS[0]]
            cav_m2, cld_m2, cav_p2, cld_p2 = row[CAMERAS[1]]

            # Pad delle immagini originali alla STESSA dimensione della maschera
            if img1_raw is not None and img2_raw is not None:
                img1_base = padOrigin(img1_raw, cav_m1.shape, origin1)
                img2_base = padOrigin(img2_raw, cav_m2.shape, origin2)
            else:
                img1_base = np.zeros(cav_m1.shape, dtype=np.uint8)
                img2_base = np.zeros(cav_m2.shape, dtype=np.uint8)

            # Converti in scala di grigi a BGR per poter disegnare a colori
            vis1 = cv2.cvtColor(img1_base, cv2.COLOR_GRAY2BGR) if len(img1_base.shape) == 2 else img1_base.copy()
            vis2 = cv2.cvtColor(img2_base, cv2.COLOR_GRAY2BGR) if len(img2_base.shape) == 2 else img2_base.copy()

            # --- RENDER CAMERA 1 ---
            vis1 = apply_overlay(vis1, cav_m1, COLOR_CAVITY, ALPHA)
            vis1 = apply_overlay(vis1, cld_m1, COLOR_CLOUD, ALPHA)
            vis1 = draw_points(vis1, cav_p1, COLOR_PTS_CAVITY)
            vis1 = draw_points(vis1, cld_p1, COLOR_PTS_CLOUD)
            cv2.putText(vis1, f"Cam1 | Key: {key} | Match: {idx}/{len(df)-1}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # --- RENDER CAMERA 2 ---
            vis2 = apply_overlay(vis2, cav_m2, COLOR_CAVITY, ALPHA)
            vis2 = apply_overlay(vis2, cld_m2, COLOR_CLOUD, ALPHA)
            vis2 = draw_points(vis2, cav_p2, COLOR_PTS_CAVITY)
            vis2 = draw_points(vis2, cld_p2, COLOR_PTS_CLOUD)
            cv2.putText(vis2, f"Cam2 | Key: {key} | Match: {idx}/{len(df)-1}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Unisci le immagini fianco a fianco
            combined = np.hstack((vis1, vis2))
            
            # Ridimensiona per farla stare nello schermo
            combined_resized = resize_to_fit(combined, MAX_SCREEN_W, MAX_SCREEN_H)

            # Mostra l'immagine
            cv2.imshow("Dataframe Scope", combined_resized)

            # Gestione dei comandi tastiera (e logica di persistenza di 500ms)
            wait_time = 0 if paused else DELAY_MS
            keypress = cv2.waitKey(wait_time) & 0xFF
            
            if keypress == ord('q'):  # Esci
                if f_h5: f_h5.close()
                cv2.destroyAllWindows()
                print("Uscita dallo scope.")
                return
            elif keypress == ord('p'):  # Pausa / Riprendi
                paused = not paused

    if f_h5: f_h5.close()
    cv2.destroyAllWindows()
    print("Esplorazione completata.")

if __name__ == "__main__":
    run_scope()