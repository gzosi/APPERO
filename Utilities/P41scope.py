from Config import Config 
from pathlib import Path
import pickle
import sys
import h5py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE MANUALE ---
DATASET = 'Dataset1'
KEY = '00030' # Sostituisci con la chiave desiderata (deve essere una stringa se nel file è salvata come tale)
# ------------------------------

def padOrigin(img, shape_wh, start_coords_xy):
    """Funzione di padding con unpack sicuro per allineare l'immagine al sensore."""
    numpy_shape = (shape_wh[1], shape_wh[0]) 
    result = np.zeros(numpy_shape, dtype=img.dtype) 
    
    if isinstance(start_coords_xy, (list, tuple, np.ndarray, pd.Series)) and len(start_coords_xy) >= 2:
        if hasattr(start_coords_xy, 'tolist'):
            start_coords_xy = start_coords_xy.tolist()
        x_start, y_start = int(start_coords_xy[0]), int(start_coords_xy[1])
    else:
        x_start, y_start = 0, 0
        
    y_end = y_start + img.shape[0]  
    x_end = x_start + img.shape[1] 
    
    # Prevenzione slice negativi se le coordinate superano le dimensioni
    y_start = max(0, min(y_start, result.shape[0]))
    x_start = max(0, min(x_start, result.shape[1]))
    
    y_end_clamped = max(y_start, min(y_end, result.shape[0]))
    x_end_clamped = max(x_start, min(x_end, result.shape[1]))
    
    src_y_end = y_end_clamped - y_start
    src_x_end = x_end_clamped - x_start
    
    if src_y_end > 0 and src_x_end > 0:
        result[y_start:y_end_clamped, x_start:x_end_clamped] = img[0:src_y_end, 0:src_x_end]
        
    return result

def main():
    print(f"🔄 Avvio P41Scope per {DATASET} - Key {KEY}...")
    main_root = Path(Config.Paths.mainRooot)
    
    # 1. Path generati
    ROOT_PKL = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase4.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.__name__ /
        Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        
    ROOT_H5 = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot / 
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ /
        Config.Packages.Drivers.Phases.Phase0.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)

    SHAPE_ROOT = (main_root /
        Config.Paths.DataRoots.ResourcesRoot /
        Config.Paths.DataRoots.StreamRoot /
        Config.Paths.DataRoots.CaseStudyRoot() /
        Config.Packages.Drivers.__name__ / 
        Config.Packages.Drivers.Phases.Phase0.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
        Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__)

    # --- Apertura PKL ---
    try:
        with open(ROOT_PKL, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Errore durante il caricamento del file PKL: {e}")
        sys.exit(1)

    # --- Controlli Dataset e Key ---
    if DATASET not in data:
        print(f"❌ Errore: Dataset '{DATASET}' non trovato nel file.")
        sys.exit(1)

    dataset_data = data[DATASET]
    if KEY not in dataset_data:
        print(f"❌ Errore: Key scelta ({KEY}) assente.")
        print(f"Scegli tra queste: {sorted(list(dataset_data.keys()))}")
        sys.exit(1)

    frame_data = dataset_data[KEY]
    PHASE = frame_data.get('Phase', 'N/A')
    cameras = [k for k in frame_data.keys() if k != 'Phase']
    
    print(f"✅ Key trovata. Phase associata: {PHASE}. Telecamere salvate: {cameras}")

    # --- Caricamento Origins e Shape ---
    try:
        origins_file = SHAPE_ROOT / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt
        origins_data = pd.read_json(origins_file)
        shape = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.Settings.FullSensorShape
    except Exception as e:
        print(f"❌ Errore nel caricare shape e origins JSON: {e}")
        sys.exit(1)

    # --- Inizializzazione PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title('P41 Scope - Overlay Maschere')

    # --- Estrazione immagini H5 e Disegno Maschere ---
    try:
        with h5py.File(ROOT_H5, 'r') as f_h5:
            for idx, camera in enumerate(cameras[:2]):
                print(f"\n🔍 Processando telecamera: {camera}")
                
                # 1. Carica l'immagine originale
                try:
                    raw = f_h5[camera]['Database4'][DATASET][KEY][:].astype(np.float32)
                    
                    # Normalizzazione per visualizzazione corretta (previene immagini nere)
                    if raw.max() > 0:
                        raw = (raw / raw.max()) * 255.0
                    raw = raw.astype(np.uint8)
                    
                    print(f"  - Immagine H5 letta. Shape: {raw.shape}, Min: {raw.min()}, Max: {raw.max()}")
                except KeyError:
                    print(f"  ❌ Errore: Key mancante nell'H5. Disegno base vuota.")
                    raw = np.zeros(shape[::-1], dtype=np.uint8)

                # 2. Pad dell'immagine originaria 
                if raw.shape[:2] != tuple(shape[::-1]):
                    try:
                        start_coords = origins_data[camera]['Database4'][DATASET]
                        raw_padded = padOrigin(raw, shape, start_coords)
                        print(f"  - Padding applicato con offset: {start_coords}")
                    except KeyError:
                        print(f"  ⚠️ Offset non trovato, l'immagine non è allineata.")
                        raw_padded = raw
                else:
                    raw_padded = raw

                # 3. Maschera dal PKL
                mask = frame_data[camera]
                if mask.max() == 1: # Assicura che la maschera sia a 255 e non a 1
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                    
                print(f"  - Maschera PKL letta. Shape: {mask.shape}, Max: {mask.max()}")

                # Verifica shape mismatch
                if raw_padded.shape[:2] != mask.shape[:2]:
                    print(f"  ⚠️ ATTENZIONE: Dimensioni sfalsate! Img={raw_padded.shape[:2]}, Mask={mask.shape[:2]}")
                    # Adatta l'immagine alla dimensione della maschera per permettere l'overlay visivo
                    raw_padded = cv2.resize(raw_padded, (mask.shape[1], mask.shape[0]))

                # 4. Disegno Overlay e Contorni (Verde Semi-Trasparente con bordo Rosso)
                display_img = cv2.cvtColor(raw_padded, cv2.COLOR_GRAY2RGB)
                overlay = display_img.copy()
                
                contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
                print(f"  - Trovati {len(contours)} contorni.")
                
                # Colora internamente di verde e fai la fusione (Alpha blending)
                # cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
                display_img = cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0)
                
                # Disegna il contorno esterno rosso acceso
                cv2.drawContours(display_img, contours, -1, (255, 0, 0), thickness=2)

                # 5. Visualizzazione nel plot
                axes[idx].imshow(display_img)
                axes[idx].set_title(f"Telecamera: {camera}")
                axes[idx].axis('off')

    except Exception as e:
        print(f"❌ Errore critico durante la lettura H5 o il plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Messaggio di compensazione per camera singola
    if len(cameras) < 2:
        axes[1].text(0.5, 0.5, 'Nessuna Seconda Telecamera', horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].axis('off')

    plt.suptitle(f"Dataset: {DATASET} | Phase: {PHASE} | Key: {KEY}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    print("\n✅ Generazione completata. Apertura della finestra (chiudila per terminare lo script)...")
    plt.show(block=True)

if __name__ == "__main__":
    main()