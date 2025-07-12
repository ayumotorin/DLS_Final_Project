import torch
from tqdm import tqdm
import numpy as np
import rasterio


def predict_sliding_window(model, full_image, patch_size=512, stride=512, device='cpu'):
    """
    Делает предсказание для большого изображения с использованием скользящего окна.
    
    :param model: Обученная модель Pytorch.
    :param full_image: Изображение в формате numpy array (H, W, C).
    :param patch_size: Размер патча.
    :param stride: Шаг, с которым двигается окно. Меньше patch_size для перекрытия.
    :param device: Устройство для вычислений ('cuda' или 'cpu').
    :return: Финальная маска предсказаний (H, W).
    """
    model.eval()
    H, W, _ = full_image.shape
    
    full_image_float = full_image.astype(np.float32)
    
    prediction_map = np.zeros((H, W), dtype=np.float32)
    counts_map = np.zeros((H, W), dtype=np.float32)

    for y in tqdm(range(0, H, stride), desc="Sliding Window Inference"):
        for x in range(0, W, stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            
            patch = full_image_float[y:y_end, x:x_end]
            
            h_patch, w_patch, _ = patch.shape
            padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            padded_patch[:h_patch, :w_patch] = patch
            
            input_tensor = torch.from_numpy(padded_patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            prediction_map[y:y_end, x:x_end] += pred_prob[:h_patch, :w_patch]
            counts_map[y:y_end, x:x_end] += 1
            

    counts_map[counts_map == 0] = 1
    final_prediction = prediction_map / counts_map
    
    return final_prediction


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние (в метрах) между двумя точками на Земле
    по их широтам и долготам.
    
    :param lat1, lon1: Широта и долгота первой точки (в градусах).
    :param lat2, lon2: Широта и долгота второй точки (в градусах).
    :return: Расстояние в метрах.
    """
    R = 6371000  # Радиус Земли в метрах
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_phi = np.deg2rad(lat2 - lat1)
    delta_lambda = np.deg2rad(lon2 - lon1)
    
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    
    return distance


def get_pixel_size_m2(tiff_path: str):
    with rasterio.open(tiff_path) as src:
        if not src.crs.is_geographic:
            transform = src.transform
            x_scale = transform.a
            y_scale = -transform.e
            if x_scale > 0 and y_scale > 0:
                pixel_area = x_scale * y_scale
                return pixel_area
    
        lon1, lat1, lon2, lat2 = src.bounds
        width = haversine_distance(lat1, lon1, lat1, lon2)
        height = haversine_distance(lat1, lon1, lat2, lon1)
        xscale = width/src.width
        yscale = height/src.height
        return xscale*yscale
    

def calculate_building_area(pred_mask_binary, scale):
    return np.sum(pred_mask_binary)*scale