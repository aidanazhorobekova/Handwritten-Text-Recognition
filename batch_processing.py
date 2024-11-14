import os
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from line_segmentation import segment_lines
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Загрузка CSV файлов с метками
train_labels = pd.read_csv('written_name_train_v2.csv')
test_labels = pd.read_csv('written_name_test_v2.csv')
validation_labels = pd.read_csv('written_name_validation_v2.csv')

# Объединение всех меток для кодирования
all_labels = pd.concat([train_labels['IDENTITY'], test_labels['IDENTITY'], validation_labels['IDENTITY']])
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Кодирование меток
train_labels['ENCODED_IDENTITY'] = label_encoder.transform(train_labels['IDENTITY'])
test_labels['ENCODED_IDENTITY'] = label_encoder.transform(test_labels['IDENTITY'])
validation_labels['ENCODED_IDENTITY'] = label_encoder.transform(validation_labels['IDENTITY'])

def get_text_from_csv(filename, labels_df):
    """Получает истинный текст из CSV на основе имени файла."""
    row = labels_df[labels_df['FILENAME'] == filename]
    if not row.empty:
        return row['ENCODED_IDENTITY'].values[0]
    else:
        return "Not found"

# Загрузка модели и процессора TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Функция распознавания текста
def ocr_handwritten(image, processor, model):
    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Данные и метки
datasets = {
    "train_v2": train_labels,
    "test_v2": test_labels,
    "validation_v2": validation_labels,
}

# Файл для записи результатов
with open("results.txt", "w") as f_out:
    for folder, labels_df in datasets.items():
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder, filename)
                try:
                    # Чтение и сегментация изображения
                    image_cv2 = cv2.imread(image_path)
                    if image_cv2 is None:
                        print(f"Не удалось загрузить изображение {image_path}")
                        continue

                    segmented_lines = segment_lines(image_cv2)
                    if not segmented_lines:
                        print(f"Не удалось сегментировать строки для {image_path}")
                        continue

                    # Получение истинного текста
                    ground_truth_text = get_text_from_csv(filename, labels_df)

                    # OCR для каждой строки
                    recognized_texts = []
                    for line in segmented_lines:
                        line_pil = Image.fromarray(line)
                        recognized_text = ocr_handwritten(line_pil, processor, model)
                        recognized_texts.append(recognized_text)
                        print(f"Recognized text: {recognized_text}")

                    # Запись в файл
                    f_out.write(f"{filename} ({folder}): Ground truth text: {ground_truth_text}, Recognized text: {' '.join(recognized_texts)}\n")

                except Exception as e:
                    print(f"Ошибка при обработке файла {filename}: {e}")
