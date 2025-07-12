import streamlit as st
import torch
import cv2
import numpy as np
import rasterio
from PIL import Image
from utils import predict_sliding_window, get_pixel_area
import segmentation_models_pytorch as smp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './model.pth'
TEST_IMAGE_PATH = './data/val/images/vienna11.tif'
TRUE_MASK_PATH = './data/val/gt/vienna11.tif'
PATCH_SIZE = 512  # Размер патча, на котором обучалась модель
STRIDE = 512
THRESHOLD = 0.5


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
            pixel_area = get_pixel_area("temp_image.tif")
        
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
            st.image(image_rgb, caption='Оригинальное изображение', use_column_width=True)
        
        with col2:
            st.image(binary_mask * 255, caption='Предсказанная маска застройки', use_column_width=True)
            
    except Exception as e:
        st.error(f"Произошла ошибка во время обработки: {e}")

else:
    st.info("Пожалуйста, загрузите файл для начала анализа.")