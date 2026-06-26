#%% Importing Libreries
from pathlib import Path
import numpy as np
import cv2 as cv
import h5py
import torch
from segment_anything_hq import SamPredictor, sam_model_registry
from controlnet_aux import HEDdetector
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def getROI(param, idx, shape):
    """
    Definisce una ROI dinamica e periodica interpolando i punti.
    """
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
def getBetterImg(img, conf):
    """
    Migliora il contrasto e la nitidezza dell'immagine leggendo i parametri dal config.
    """
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    denoised = cv.bilateralFilter(
        img, 
        d=conf.Bilateral.d, 
        sigmaColor=conf.Bilateral.sigmaColor, 
        sigmaSpace=conf.Bilateral.sigmaSpace)
    clahe = cv.createCLAHE(
        clipLimit=conf.CLAHE.clipLimit, 
        tileGridSize=conf.CLAHE.tileGridSize)
    contrasted = clahe.apply(denoised)
    blur = cv.GaussianBlur(contrasted,
        conf.UnsharpMask.kernel, conf.UnsharpMask.sigma)
    sharpened = cv.addWeighted(
        contrasted, 
        1.0 + conf.UnsharpMask.strength, 
        blur, 
        -conf.UnsharpMask.strength, 
        0)
    return sharpened
def focusAttention(raw_F, raw_B, roi, config, edge_model=None):
    cfg = config.Focus
    enhanced_F = getBetterImg(raw_F, config.Enhancement)
    enhanced_B = getBetterImg(raw_B, config.Enhancement)
    diff = cv.subtract(enhanced_F, enhanced_B)
    edge = np.zeros_like(enhanced_F)
    if edge_model is not None:
        edge_F = np.array(edge_model(
            cv.cvtColor(enhanced_F, cv.COLOR_GRAY2RGB), 
            detect_resolution=cfg.HED.detect_resolution, scribble=False))
        edge_B = np.array(edge_model(
            cv.cvtColor(enhanced_B, cv.COLOR_GRAY2RGB), 
            detect_resolution=cfg.HED.detect_resolution, scribble=False))
        edge = cv.resize(
            cv.subtract(
                cv.cvtColor(edge_F, cv.COLOR_RGB2GRAY),
                cv.cvtColor(edge_B, cv.COLOR_RGB2GRAY)),
            (enhanced_F.shape[1], enhanced_F.shape[0]),
            interpolation=cv.INTER_LINEAR)
    entr_raw = entropy(cv.convertScaleAbs(raw_F), disk(cfg.Entropy.disk_size))
    entr = np.clip((entr_raw / 8.0) * 255, 0, 255).astype(np.uint8)
    f1 = entr.astype(np.float32) / 255.0
    f2 = edge.astype(np.float32) / 255.0
    f3 = diff.astype(np.float32) / 255.0
    w = cfg.Weights
    attention_map = (f1 * w.entropy) + (f2 * w.hed) + (f3 * w.diff)
    attention_map = np.clip(attention_map * 255, 0, 255).astype(np.uint8)
    if roi is not None:
        attention_map = cv.bitwise_and(attention_map, attention_map, mask=roi)
        roi_pixels = attention_map[roi > 0]
        if len(roi_pixels) > 0:
            thresh_val, _ = cv.threshold(roi_pixels.reshape(-1, 1), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            _, binary = cv.threshold(attention_map, thresh_val, 255, cv.THRESH_BINARY)
            binary = cv.bitwise_and(binary, binary, mask=roi)
        else:
            binary = np.zeros_like(attention_map)
    else:
        _, binary = cv.threshold(attention_map, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    openKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.Morph.Kernels.Open)
    dilateKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.Morph.Kernels.Dilate)
    erodeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.Morph.Kernels.Erode)    
    openBinary = cv.morphologyEx(binary, cv.MORPH_OPEN, openKernel)
    canny = cv.Canny(openBinary, 50, 150)
    dilateCanny = cv.dilate(canny, dilateKernel, iterations=cfg.Morph.Its.Dilate)
    contours, _ = cv.findContours(dilateCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(binary)
    min_area = cfg.Morph.AreaMin
    for contour in contours:
        area = cv.contourArea(contour)
        if area > min_area:
            cv.drawContours(final_mask, [contour], -1, 255, cv.FILLED)
    focus = cv.erode(final_mask, erodeKernel, iterations=cfg.Morph.Its.Erode)
    sanity = cv.absdiff(
        cv.bitwise_and(raw_F, raw_F, mask=focus),
        cv.bitwise_and(raw_B, raw_B, mask=focus))
    nonZero = sanity[sanity > 0]
    if len(nonZero) > 0:
        p = np.percentile(nonZero, 100-cfg.EmptyCheck.TopPercent) 
        topDiff = nonZero[nonZero >= p]
        if np.mean(topDiff) < cfg.EmptyCheck.MeanThresh:
            focus = np.zeros_like(focus)
    return focus
def getSmartPrompt(focusMap, config):
    """
    Genera smart prompt (BBox, Punti Positivi Multipli, Punti Negativi) per SAM
    partendo da una maschera binaria. Usa la distanza dai bordi per generare 
    prompt concentrici e K-Means per distribuire uniformemente i punti positivi.
    """
    min_area = config.MinArea
    num_target_points = config.TargetPts
    negative_ring = config.NegativeRing
    percentile_levels = config.PercentileLevels
    iou_threshold = config.IouThreshold
    def get_iou(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection_area / float(box1_area + box2_area - intersection_area)
    if cv.countNonZero(focusMap) == 0:
        return []
    found_objects = []
    img_height, img_width = focusMap.shape
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dist_global = cv.distanceTransform(focusMap, cv.DIST_L2, 5)
    active_pixels = dist_global[dist_global > 0]
    if len(active_pixels) == 0:
        return []
    thresholds = np.percentile(active_pixels, percentile_levels)
    thresholds = sorted(list(set(thresholds))) 
    for threshold in thresholds:
        _, binarized = cv.threshold(dist_global, threshold, 255, cv.THRESH_BINARY)
        binarized = binarized.astype(np.uint8) 
        binarized = cv.morphologyEx(binarized, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(binarized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            if any(get_iou(bbox, obj["bbox"]) > iou_threshold for obj in found_objects):
                continue
            local_contour = contour - [x, y] 
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv.drawContours(local_mask, [local_contour], -1, 255, thickness=cv.FILLED)
            local_dist = cv.distanceTransform(local_mask, cv.DIST_L2, 5)
            max_val = np.max(local_dist)
            safe_thresh = max(max_val * 0.3, 2.0) if max_val > 2.0 else max_val
            safe_y, safe_x = np.where(local_dist >= safe_thresh)
            safe_pts = np.column_stack((safe_x, safe_y)).astype(np.float32)
            positive_points = []
            if len(safe_pts) >= num_target_points and num_target_points > 1:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, _, centers = cv.kmeans(safe_pts, num_target_points, None, criteria, 10, cv.KMEANS_PP_CENTERS)
                for center in centers:
                    dists = np.linalg.norm(safe_pts - center, axis=1)
                    closest_idx = np.argmin(dists)
                    best_pt = safe_pts[closest_idx]
                    pt_x, pt_y = int(best_pt[0]) + x, int(best_pt[1]) + y
                    positive_points.append([pt_x, pt_y])
                unique_pts = list(dict.fromkeys([(pt[0], pt[1]) for pt in positive_points]))
                positive_points = [[pt[0], pt[1]] for pt in unique_pts]
            else:
                _, _, _, max_loc_local = cv.minMaxLoc(local_dist)
                positive_points.append([max_loc_local[0] + x, max_loc_local[1] + y])

            dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (negative_ring, negative_ring))
            dilated_local = cv.dilate(local_mask, dilate_kernel)
            annulus = cv.subtract(dilated_local, local_mask)
            neg_y, neg_x = np.where(annulus > 0)
            negative_points = []
            if len(neg_y) >= 4:
                indices = np.linspace(0, len(neg_y) - 1, 4, dtype=int)
                for idx in indices:
                    negative_points.append([neg_x[idx] + x, neg_y[idx] + y])
            else:
                nx1, ny1 = max(0, x - negative_ring), max(0, y - negative_ring)
                nx2, ny2 = min(img_width-1, x + w + negative_ring), max(0, y - negative_ring)
                nx3, ny3 = max(0, x - negative_ring), min(img_height-1, y + h + negative_ring)
                nx4, ny4 = min(img_width-1, x + w + negative_ring), min(img_height-1, y + h + negative_ring)
                negative_points = [[nx1, ny1], [nx2, ny2], [nx3, ny3], [nx4, ny4]]
            point_coords = positive_points + negative_points
            point_labels = [1] * len(positive_points) + [0, 0, 0, 0]
            found_objects.append({
                "area": area,
                "bbox": bbox,
                "point_coords": point_coords,
                "point_labels": point_labels
            })
    found_objects.sort(key=lambda item: item["area"], reverse=True)
    sam_prompts = [{
        "bbox": obj["bbox"],
        "point_coords": obj["point_coords"],
        "point_labels": obj["point_labels"]
    } for obj in found_objects]
    return sam_prompts
def segmentEngine(img, predictor, prompts):
    masks = []
    predictor.set_image(cv.cvtColor(img, cv.COLOR_GRAY2BGR))  
    for prompt in prompts:
        input_box = np.array(prompt['bbox'])
        input_points = np.array(prompt['point_coords'])
        input_labels = np.array(prompt['point_labels'])
        mask, scores, _ = predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            box=input_box[None, :],
            multimask_output=True)
        masks.append(mask[np.argmax(scores)])
    return masks
def groupMasks(masks, attentionMap, roi, config):
    """
    1. Taglia maschere in base a ROI.
    2. Raggruppa per IoU usando DFS nativo.
    3. Assegna Score calcolato tramite Intersection over Union (IoU) con l'attentionMap.
    """
    roi_bool, att_bool = roi > 0, attentionMap > 0 
    objects = []
    for mask in masks:
        mask_in_roi = (mask > 0) & roi_bool
        if not mask_in_roi.any(): 
            continue
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_in_roi.astype(np.uint8) * 255)
        objects.extend([labels == i for i in range(1, num_labels) if stats[i, cv.CC_STAT_AREA] >= config.AreaMin])
    n = len(objects)  
    if n == 0: return [], []
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            inter = (objects[i] & objects[j]).sum()
            union = (objects[i] | objects[j]).sum()
            if union > 0 and (inter / union) >= config.Similarity:
                adj_matrix[i, j] = adj_matrix[j, i] = True            
    visited, connected_labels, current_label = np.zeros(n, dtype=bool), np.zeros(n, dtype=int), 0
    for i in range(n):
        if not visited[i]:
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    connected_labels[node] = current_label
                    stack.extend(np.where(adj_matrix[node])[0])
            current_label += 1
    groups = [[] for _ in range(current_label)]
    for idx, (label, mask) in enumerate(zip(connected_labels, objects)):
        intersection = (mask & att_bool).sum()
        union = (mask | att_bool).sum()
        score = intersection / float(union) if union > 0 else 0.0
        groups[label].append({'id': idx, 'score': score})
    for g in groups:
        g.sort(key=lambda x: x['score'], reverse=True)
    return groups, objects
def maskCollapse(groups, expanded_masks, roi, collapse_config):
    """
    Estrae le maschere Rank 1, calcola una soglia dinamica basata sul percentile
    e su una percentuale del valore massimo, e fonde i vincitori.
    """
    bool_roi = roi > 0
    if not groups:
        return [], np.zeros_like(bool_roi)
    rank1 = [g[0] for g in groups if len(g) > 0]
    if not rank1:
        return [], np.zeros_like(bool_roi)
    scores = np.array([info['score'] for info in rank1])
    relative_thresh = np.percentile(scores, collapse_config.PercentileThresh)
    max_score = np.max(scores)
    percentage_thresh = max_score * (collapse_config.MinMaxPercentage / 100.0)
    dynamic_thresh = max(relative_thresh, percentage_thresh)
    union = np.zeros_like(bool_roi)
    for info in rank1:
        if info['score'] >= dynamic_thresh:
            mask_id = info['id']
            obj_mask = expanded_masks[mask_id]
            union |= obj_mask
    return [union], union # qui forse ci rimettero le mani ma per il momento lo teniamo cosi (a maschera unica)
def cloudSegmenter(img, mask, config):
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8), 0
    cv_mask = (mask > 0).astype(np.uint8) * 255
    blurred = cv.GaussianBlur(img, tuple(config.BlurKernel), 0) 
    masked_pixels = blurred[mask > 0].reshape(-1, 1) 
    val, _ = cv.threshold(masked_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    relaxed = val * config.Relaxation 
    _, binary = cv.threshold(blurred, relaxed, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, tuple(config.DilateKernel))
    expanded_mask = cv.dilate(cv_mask, kernel, iterations=config.DilateIter)
    region = cv.bitwise_and(binary, expanded_mask)
    _, labels = cv.connectedComponents(region)
    overlapping_labels = np.unique(labels[mask > 0])
    valid_labels = overlapping_labels[overlapping_labels > 0]
    cloud = np.isin(labels, valid_labels).astype(np.uint8) * 255
    return cloud
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1
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
        dstRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase3.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.__name__)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        period = ppr / blades 
        bounds = settings.Bounds
        model = sam_model_registry[
            task_conf.Settings.Segmenter.Name](
                checkpoint = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.DependeciesRoot / 
            task_conf.Settings.Segmenter.Model /
            task_conf.Settings.Segmenter.Checkpoint))
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(model)
        edge_model = HEDdetector.from_pretrained("lllyasviel/Annotators")
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = list(f.keys())
                groupsF = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
                    for camera in cameras}
                groupsB = {camera:
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Background]
                    for camera in cameras} 
                proces = [k for k in groupsF[cameras[0]].keys()
                    if any(lower <= (int(k) % period) <= upper for lower, upper in bounds)]
                for key in tqdm(proces, total=len(proces), desc=colored(f'Cavitation Analysis 🚀', 'magenta'), ncols=100):
                    data = dict()
                    phase = int(key) % period
                    for camera in cameras:
                        try:  
                            raw_F = groupsF[camera][key][:].astype(np.uint8)
                            raw_B = groupsB[camera][key][:].astype(np.uint8)
                            roi, _ = getROI(
                                getattr(task_conf.Settings.DynamicROI, camera), phase, raw_F.shape)
                            focusMap = focusAttention(raw_F, raw_B, roi, task_conf.Settings, edge_model=edge_model)
                            prompts = getSmartPrompt(focusMap, task_conf.Settings.SmartPrompt)
                            if len(prompts) > 0:
                                masks = segmentEngine(raw_F, predictor, prompts)
                                groups, objects = groupMasks(
                                    masks, focusMap, roi, task_conf.Settings.Group)
                                cavities, collapse = maskCollapse(groups, objects, roi, task_conf.Settings.Collapse)
                                cloud = cloudSegmenter(raw_F, collapse, task_conf.Settings.Cloud)
                            else:
                                cavities = [np.zeros_like(raw_F)]
                                collapse = np.zeros_like(raw_F)
                                cloud = np.zeros_like(raw_F)
                            data[camera] = [cavities, cloud, roi]  
                        except Exception as e:
                            data[camera] = [[np.zeros_like(raw_F)], np.zeros_like(raw_F), np.zeros_like(raw_F)] 
                    dst = (dstRoot / f'{task_conf.MetaData.OutputName}_{key}{task_conf.MetaData.OutputExt}')
                    pickle.dump(data, open(dst, 'wb'))
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return 