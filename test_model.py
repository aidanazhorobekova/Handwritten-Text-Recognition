import os
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image  # Для работы с изображениями, так как TrOCR использует PIL форматы

# Загрузка модели и процессора из сохраненной директории
processor = TrOCRProcessor.from_pretrained("./saved_model_handwritten")
model = VisionEncoderDecoderModel.from_pretrained("./saved_model_handwritten")

def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")  # Открываем изображение с помощью PIL
    return image

# Папка с тестовыми изображениями
test_data_path = 'test_v2'

for filename in os.listdir(test_data_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(test_data_path, filename)
        try:
            # Загрузка и предобработка изображения
            sample_image = load_and_preprocess_image(image_path)
            
            # Преобразуем изображение для подачи в TrOCR модель
            pixel_values = processor(sample_image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"Recognized text for {filename}: {recognized_text}")
        except FileNotFoundError as e:
            print(e)
