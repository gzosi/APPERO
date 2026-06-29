#%% Importing Libreries
from pathlib import Path
import numpy as np
import pandas as pd
import cv2 as cv
import h5py, torch, pickle
from segment_anything_hq import SamPredictor, sam_model_registry
from tqdm import tqdm
from termcolor import colored
#%% Defining Subroutines
def padOrigin(img, shape_wh, start_coords_xy):
    numpy_shape = (shape_wh[1], shape_wh[0]) 
    result = np.zeros(numpy_shape, dtype=img.dtype) 
    x_start, y_start = start_coords_xy
    y_end = y_start + img.shape[0]  
    x_end = x_start + img.shape[1] 
    y_end_clamped = min(y_end, result.shape[0])
    x_end_clamped = min(x_end, result.shape[1])
    src_y_end = y_end_clamped - y_start
    src_x_end = x_end_clamped - x_start
    result[y_start:y_end_clamped, x_start:x_end_clamped] = img[0:src_y_end, 0:src_x_end]
    return result
def getROI(param, idx, shape):
    indices = sorted(param.keys())
    if not indices:
        return np.zeros(shape[:2], dtype=np.uint8), None
    if idx <= indices[0]:
        target_pts = np.array(param[indices[0]], dtype=np.float32)
    elif idx >= indices[-1]:
        target_pts = np.array(param[indices[-1]], dtype=np.float32)
    else:
        target_pts = None
        for i in range(len(indices) - 1):
            t0 = indices[i]
            t1 = indices[i+1]
            if t0 <= idx <= t1:
                alpha = (idx - t0) / (t1 - t0)
                p0 = np.array(param[t0], dtype=np.float32)
                p1 = np.array(param[t1], dtype=np.float32)
                target_pts = p0 + (p1 - p0) * alpha
                break
    mask = np.zeros(shape[:2], dtype=np.uint8)
    int_pts = target_pts.astype(np.int32)
    cv.fillPoly(mask, [int_pts], 255)
    return mask, int_pts
def focusAttention(raw_F, mu, sigma, roi_padded, settings):
    epsilon = settings.Zscore.epsilon
    zTh = settings.Zscore.zTh
    diff = raw_F.astype(np.float32) - mu
    diff_positive = np.maximum(diff, 0)
    z_score_map = diff_positive / (sigma + epsilon)
    z_score_map[roi_padded == 0] = 0
    base_anomaly_mask = (z_score_map > zTh).astype(np.uint8) * 255
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, settings.Zscore.Morph.openKernel)
    clean_anomaly_mask = cv.morphologyEx(base_anomaly_mask, cv.MORPH_OPEN, open_kernel)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, settings.Zscore.Morph.closeKernel)
    fused_anomaly_mask = cv.morphologyEx(clean_anomaly_mask, cv.MORPH_CLOSE, close_kernel)
    return fused_anomaly_mask, clean_anomaly_mask
def getSmartPrompt(focusMap, config):
    if cv.countNonZero(focusMap) == 0:
        return []
    img_height, img_width = focusMap.shape
    contours, _ = cv.findContours(focusMap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sam_prompts = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < config.MinArea:
            continue  
        x, y, w, h = cv.boundingRect(contour)
        max_dim = max(w, h)
        dynamic_pad = max(config.BboxPaddingBase, int(max_dim * 0.15))
        dynamic_ring = max(config.NegativeRingBase, int(max_dim * 0.20))
        x1, y1 = max(0, x - dynamic_pad), max(0, y - dynamic_pad)
        x2, y2 = min(img_width, x + w + dynamic_pad), min(img_height, y + h + dynamic_pad)
        bbox = [x1, y1, x2, y2]
        local_contour = contour - [x, y] 
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(local_mask, [local_contour], -1, 255, thickness=cv.FILLED)
        local_dist = cv.distanceTransform(local_mask, cv.DIST_L2, 3)
        max_dist = np.max(local_dist)
        safe_thresh = max(max_dist * 0.5, 1.0)
        safe_y, safe_x = np.where(local_dist >= safe_thresh)
        safe_pts = np.column_stack((safe_x, safe_y)).astype(np.float32)
        positive_points = []
        if len(safe_pts) >= config.TargetPts and config.TargetPts > 1:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv.kmeans(safe_pts, config.TargetPts, None, criteria, 10, cv.KMEANS_PP_CENTERS)
            for center in centers:
                positive_points.append([int(center[0]) + x, int(center[1]) + y])
        elif len(safe_pts) > 0:
            _, _, _, max_loc_local = cv.minMaxLoc(local_dist)
            positive_points.append([max_loc_local[0] + x, max_loc_local[1] + y])
        unique_pts = list(dict.fromkeys([(pt[0], pt[1]) for pt in positive_points]))
        positive_points = [[pt[0], pt[1]] for pt in unique_pts]
        kernel_neg = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dynamic_ring, dynamic_ring))
        dilated_local = cv.dilate(local_mask, kernel_neg)
        annulus = cv.subtract(dilated_local, local_mask) 
        neg_y, neg_x = np.where(annulus > 0)
        negative_points = []
        if len(neg_y) > config.NumNegativePts:
            indices = np.linspace(0, len(neg_y) - 1, config.NumNegativePts, dtype=int)
            for idx in indices:
                negative_points.append([int(neg_x[idx]) + x, int(neg_y[idx]) + y])
        elif len(neg_y) > 0:
            for nx, ny in zip(neg_x, neg_y):
                negative_points.append([int(nx) + x, int(ny) + y])
        point_coords = positive_points + negative_points
        point_labels = [1] * len(positive_points) + [0] * len(negative_points) 
        if len(point_coords) > 0:
            sam_prompts.append({
                "bbox": bbox,
                "point_coords": point_coords,
                "point_labels": point_labels})
    return sam_prompts
def segmentEngine(img, predictor, prompts):
    masks = []
    img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    predictor.set_image(img_rgb)  
    for prompt in prompts:
        input_box = np.array(prompt['bbox'])
        input_points = np.array(prompt['point_coords'])
        input_labels = np.array(prompt['point_labels'])
        mask, _, _ = predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            box=input_box[None, :],
            multimask_output=False)
        masks.append(mask[0]) 
    return masks
#%% Defining Main Function
def main(Config): 
    task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
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
        shapeRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot /
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase0.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__)
        dstRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase4.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.__name__)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        period, bounds = int(ppr / blades), settings.Bounds
        model = sam_model_registry[task_conf.Settings.Segmenter.Name](
            checkpoint = (main_root /
                Config.Paths.DataRoots.ResourcesRoot /
                Config.Paths.DataRoots.DependeciesRoot / 
                task_conf.Settings.Segmenter.Model /
                task_conf.Settings.Segmenter.Checkpoint))
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(model)
        try:
            origins_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt)
            cameras = list(origins_data.keys())
            shape = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.Settings.FullSensorShape
            aggregated = {}
            with h5py.File(srcRoot, 'r') as f:
                datasets = sorted([dat for dat in f[cameras[0]][settings.Src.Database].keys()],
                    key=lambda x: int(x.replace('Dataset', '')))
                for camera in cameras:
                    ref_keys = [k for k in f[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Location].keys()]
                    for dataset in datasets:
                        if dataset not in aggregated:
                            aggregated[dataset] = {}
                        cache = {}
                        for phase in range(period):
                            ref_phase = [k for k in ref_keys if abs(int(k) - phase) % ppr == 0]
                            if not ref_phase: continue
                            stack_B = np.stack([
                                padOrigin(
                                    f[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Location][k][:].astype(np.uint8),
                                    shape, 
                                    origins_data[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Location]
                                ) for k in ref_phase], axis=0)
                            mu = np.mean(stack_B, axis=0).astype(np.float32)
                            sigma = np.std(stack_B, axis=0).astype(np.float32)
                            cache[phase] = {'mu': mu, 'sigma': sigma}
                        process = [k for k in f[camera][settings.Src.Database][dataset].keys()
                        if any(lower <= (int(k) % period) <= upper for lower, upper in bounds)]
                        for key in tqdm(process, total=len(process), desc=colored(f'Damage Assessment 🚀', 'magenta'), ncols=100):
                            phase = int(key) % period
                            if phase not in cache: continue
                            raw = f[camera][settings.Src.Database][dataset][key][:].astype(np.uint8)
                            raw_F = padOrigin(raw, shape, origins_data[camera][settings.Src.Database][dataset])
                            roi_mask_raw, _ = getROI(getattr(task_conf.Settings.DynamicROI, camera), phase, raw.shape)
                            roi_padded = padOrigin(roi_mask_raw, shape, origins_data[camera][settings.Src.Database][dataset])
                            mu = cache[phase]['mu']
                            sigma = cache[phase]['sigma']
                            fused_anomaly_mask, clean_anomaly_mask = focusAttention(raw_F, mu, sigma, roi_padded, settings)
                            sam_prompts = getSmartPrompt(fused_anomaly_mask, settings.SmartPrompt)
                            sam_raw_mask = np.zeros_like(raw_F, dtype=np.uint8)
                            if len(sam_prompts) > 0:
                                predicted_masks = segmentEngine(raw_F, predictor, sam_prompts)
                                for m in predicted_masks:
                                    mask_np = (m.astype(np.uint8) * 255)
                                    sam_raw_mask = cv.bitwise_or(sam_raw_mask, mask_np)
                            hybrid_mask = cv.bitwise_or(sam_raw_mask, clean_anomaly_mask)
                            containment_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, settings.ContainmentZone.Kernel)
                            containment_zone = cv.dilate(fused_anomaly_mask, containment_kernel)                
                            final_damage_mask = cv.bitwise_and(hybrid_mask, containment_zone)
                            final_damage_mask = cv.bitwise_and(final_damage_mask, final_damage_mask, mask=roi_padded)
                            clean_final_mask = np.zeros_like(final_damage_mask)
                            contours, _ = cv.findContours(final_damage_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if cv.contourArea(cnt) >=  settings.ContainmentZone.minArea:
                                    cv.drawContours(clean_final_mask, [cnt], -1, 255, thickness=cv.FILLED)       
                            final_damage_mask = clean_final_mask
                            if key not in aggregated[dataset]:
                                aggregated[dataset][key] = {'Phase': phase}
                            aggregated[dataset][key][camera] = final_damage_mask
            dst = (dstRoot / f'{task_conf.MetaData.OutputExt}')
            pickle.dump(aggregated, open(dst, 'wb'))
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return