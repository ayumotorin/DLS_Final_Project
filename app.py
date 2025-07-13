import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
import numpy as np
import rasterio


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './model.pth'
PATCH_SIZE = 512  # Размер патча, на котором обучалась модель
STRIDE = 512
THRESHOLD = 0.5


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


model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)

st.set_page_config(
    page_title="Анализ площади застройки",
    page_icon="🏘️",
    layout="wide"
)

st.title('🏘️ Калькулятор площади застройки по спутниковому снимку')
st.write("Загрузите геопривязанный TIFF-файл, чтобы получить предсказанную маску застройки и общую площадь в квадратных метрах.")


@st.cache_resource
def load_model(model_path, device):

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

DEVICE = torch.device("cpu")
try:
    with st.spinner('Загрузка модели... Это может занять некоторое время.'):
        model = load_model('model.pth', DEVICE)
    st.success('Модель успешно загружена!')
except Exception as e:
    st.error(f"Ошибка при загрузке модели: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Выберите GeoTIFF файл (*.tif, *.tiff)", 
    type=['tif', 'tiff']
)


if uploaded_file is not None:
    with open("temp_image.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Файл загружен. Начинается обработка...")
    
    try:
        # --- Шаг 1: Расчет GSD и площади пикселя ---
        with st.spinner('Анализ геопривязки...'):
            pixel_area = get_pixel_size_m2("temp_image.tif")
        
        if pixel_area is None:
            st.error("Не удалось определить масштаб из метаданных файла. Пожалуйста, убедитесь, что файл является геопривязанным GeoTIFF.")
            st.stop()
        
        st.success(f"Масштаб определен. Площадь одного пикселя: {pixel_area:.4f} м².")
        
        # --- Шаг 2: Предсказание маски ---
        image = cv2.imread("temp_image.tif")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with st.spinner('Выполняется сегментация... Этот процесс может быть долгим.'):
            # Параметры для sliding window
            PATCH_SIZE = 512
            STRIDE = 256
            THRESHOLD = 0.5
            
            pred_probs = predict_sliding_window(model, image_rgb, PATCH_SIZE, STRIDE, DEVICE)
            binary_mask = (pred_probs > THRESHOLD).astype(np.uint8)

        st.success("Сегментация завершена!")

        # --- Шаг 3: Вычисление и вывод результатов ---
        building_pixel_count = np.sum(binary_mask)
        total_area = building_pixel_count * pixel_area

        st.subheader("Результаты анализа")
        st.metric(label="Общая площадь застройки", value=f"{total_area:,.2f} м²")
        
        # --- Шаг 4: Визуализация ---
        st.subheader("Визуализация")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_rgb, caption='Оригинальное изображение', use_container_width=True)
        
        with col2:
            st.image(binary_mask * 255, caption='Предсказанная маска застройки', use_container_width=True)
            
    except Exception as e:
        st.error(f"Произошла ошибка во время обработки: {e}")

else:
    st.info("Пожалуйста, загрузите файл для начала анализа.")