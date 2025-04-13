import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_cars(model, image_rgb):
    """Выполняет детекцию автомобилей с помощью модели YOLO."""
    results = model(image_rgb)
    cars = []
    masks = []
    for result in results:
        for box, mask in zip(result.boxes, result.masks or []):
            if int(box.cls) == 2:  # Класс 2 в COCO — автомобиль
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cars.append((x1, y1, x2, y2))
                mask_data = mask.data.cpu().numpy().squeeze()
                masks.append(mask_data)
    return cars, masks


def process_car(image_rgb, car_coords, mask_data):
    """
    Обрабатывает один автомобиль:
    сегментирует, исключает стёкла/колёса, вычисляет цвет.

    """
    x1, y1, x2, y2 = car_coords
    # Выделяем область автомобиля
    car_region = image_rgb[y1:y2, x1:x2].copy()

    # Изменение размера маски до размеров bounding box
    mask_resized = cv2.resize(
        mask_data, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    # Применение маски сегментации
    car_segmented = cv2.bitwise_and(car_region, car_region, mask=mask_resized)

    # Преобразование в HSV
    car_hsv = cv2.cvtColor(car_segmented, cv2.COLOR_RGB2HSV)

    # Маскирование стёкол и колёс
    lower_glass = np.array([0, 0, 150])  # Высокая яркость
    upper_glass = np.array([180, 50, 255])  # Низкая насыщенность
    lower_wheels = np.array([0, 0, 0])  # Тёмные области
    upper_wheels = np.array([180, 255, 50])  # Низкая яркость

    mask_glass = cv2.inRange(car_hsv, lower_glass, upper_glass)
    mask_wheels = cv2.inRange(car_hsv, lower_wheels, upper_wheels)
    mask_exclude = cv2.bitwise_or(mask_glass, mask_wheels)
    mask_body = cv2.bitwise_and(mask_resized, cv2.bitwise_not(mask_exclude))

    # Применение финальной маски
    car_body = cv2.bitwise_and(car_region, car_region, mask=mask_body)

    # Вычисление среднего цвета
    valid_pixels = car_body[mask_body > 0].reshape(-1, 3)
    if len(valid_pixels) > 0:
        mean_color = np.mean(valid_pixels, axis=0).astype(int)
    else:
        mean_color = np.array([128, 128, 128])

    return tuple(mean_color)


def visualize_results(image_rgb, cars, rgb_colors):
    """Визуализация."""
    # Копия изображения для рисования
    vis_image = image_rgb.copy()

    # Рисуем bounding box'ы и метки
    for i, (x1, y1, x2, y2) in enumerate(cars):
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_image, f"Car {i+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Визуализация
    plt.figure(figsize=(15, 5))

    # Исходное изображение
    plt.subplot(1, len(cars) + 1, 1)
    plt.imshow(vis_image)
    plt.title("Автомобили с определёнными цветами")
    plt.axis("off")

    # Цветовые патчи
    for i, color in enumerate(rgb_colors):
        plt.subplot(1, len(cars) + 1, i + 2)
        color_patch = np.ones((100, 100, 3)) * (np.array(color) / 255)
        plt.imshow(color_patch)
        plt.title(f"Авто {i+1}\nRGB: {color}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():

    # Загрузка модели (vj;)
    model = YOLO("yolo11n-seg.pt")

    # Загрузка изображения
    image_path = "task2/data/istockphoto-494522913-612x612.jpg"

    # Подготовка изображения
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Детекция автомобилей
    cars, masks = detect_cars(model, image_rgb)
    if not cars:
        print("Автомобили не найдены")
        return

    # Обработка автомобилей
    rgb_colors = []
    for i, (car_coords, mask_data) in enumerate(zip(cars, masks)):
        color = process_car(image_rgb, car_coords, mask_data)
        rgb_colors.append(color)
        print(f"Авто {i+1}: RGB = {color}")

    # Визуализация результатов
    visualize_results(image_rgb, cars, rgb_colors)


if __name__ == "__main__":
    main()
