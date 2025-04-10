import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")  # Модель YOLOv8 nano для скорости

# Загрузка изображения
image_path = "data/images.jpeg"  # Замените на путь к вашему изображению
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Детекция автомобилей
results = model(image_rgb)
cars = []
for result in results:
    for box in result.boxes:
        if int(box.cls) == 2:  # Класс 2 в COCO — это автомобиль
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cars.append((x1, y1, x2, y2))

# Список для хранения результатов
rgb_colors = []

# Обработка каждого автомобиля
for i, (x1, y1, x2, y2) in enumerate(cars):
    # Выделяем верхнюю часть (капот/крыша) — верхние 30% по высоте
    # hood_height = int((y2 - y1) * 0.3)
    # hood_region = image_rgb[y1:y1 + hood_height, x1:x2]
    hood_height = int(y2 - y1)
    hood_region = image_rgb[y1:y1 + hood_height, x1:x2]

    # Вычисляем средний цвет
    mean_color = np.mean(hood_region, axis=(0, 1)).astype(int)
    rgb_colors.append(tuple(mean_color))

    # Для визуализации сохраняем область капота
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box
    cv2.rectangle(image_rgb, (x1, y1), (x2, y1 + hood_height), (0, 255, 0), 1)  # Область капота

# Вывод RGB-кортежей
print("RGB цвета автомобилей:", rgb_colors)

# Визуализация
plt.figure(figsize=(15, 5))

# Исходное изображение с bounding box’ами и областями капота
plt.subplot(1, len(cars) + 1, 1)
plt.imshow(image_rgb)
plt.title("Автомобили и области капота")
plt.axis("off")

# Цвета для каждого автомобиля
for i, color in enumerate(rgb_colors):
    plt.subplot(1, len(cars) + 1, i + 2)
    color_patch = np.ones((100, 100, 3)) * (np.array(color) / 255)
    plt.imshow(color_patch)
    plt.title(f"Авто {i+1}\nRGB: {color}")
    plt.axis("off")

plt.tight_layout()
plt.show()