import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from PIL import Image
import gradio as gr
from ultralytics import YOLO

# Загрузка модели YOLO
model = YOLO('yolo11n-seg.pt')


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
    """Обрабатывает один автомобиль: сегментирует, исключает стекла/колеса,
    вычисляет цвет.
    """
    x1, y1, x2, y2 = car_coords
    car_region = image_rgb[y1:y2, x1:x2].copy()

    # Изменение размера маски до размеров bounding box
    mask_resized = cv2.resize(mask_data, (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    # Применение маски сегментации
    car_segmented = cv2.bitwise_and(car_region, car_region, mask=mask_resized)

    # Преобразование в HSV
    car_hsv = cv2.cvtColor(car_segmented, cv2.COLOR_RGB2HSV)

    # Маскирование стекол и колес
    lower_glass = np.array([0, 0, 150])                 # Высокая яркость
    upper_glass = np.array([180, 50, 255])              # Низкая насыщенность
    lower_wheels = np.array([0, 0, 0])                  # Темные области
    upper_wheels = np.array([180, 255, 50])             # Низкая яркость

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
    """Визуализирует результаты: возвращает изображение."""
    vis_image = image_rgb.copy()

    # Рисуем bounding box'ы и метки
    for i, (x1, y1, x2, y2) in enumerate(cars):
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_image, f'Car {i+1}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Создаем фигуру
    plt.figure(figsize=(15, 5))

    # Исходное изображение
    plt.subplot(1, len(cars) + 1, 1)
    plt.imshow(vis_image)
    plt.title('Автомобили с определенными цветами')
    plt.axis('off')

    # Цветовые патчи
    for i, color in enumerate(rgb_colors):
        plt.subplot(1, len(cars) + 1, i + 2)
        color_patch = np.ones((100, 100, 3)) * (np.array(color) / 255)
        plt.imshow(color_patch)
        plt.title(f'Авто {i+1}\nRGB: {color}')
        plt.axis('off')

    plt.tight_layout()

    # Сохранение в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)


@lru_cache(maxsize=100)
def cached_process_image(image_bytes, shape):
    """Кэширует обработку изображения."""
    image_rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    cars, masks = detect_cars(model, image_rgb)
    if not cars:
        return tuple(), tuple(), image_rgb

    rgb_colors = []
    for car_coords, mask_data in zip(cars, masks):
        color = process_car(image_rgb, car_coords, mask_data)
        rgb_colors.append(color)

    vis_image = visualize_results(image_rgb, cars, rgb_colors)
    return tuple(rgb_colors), tuple(cars), vis_image


def demo_process_image(image):
    """Обрабатывает изображение для Gradio."""
    if image is None:
        return 'Пожалуйста, загрузите изображение', None

    try:
        # Конвертация изображения в RGB
        image_rgb = np.array(image)

        # Подготовка данных для кэширования
        image_bytes = image_rgb.tobytes()
        shape = image_rgb.shape

        # Обработка с кэшированием
        rgb_colors, cars, vis_image = cached_process_image(image_bytes, shape)

        if not cars:
            return 'Автомобили не найдены', None

        # Формирование текстового результата
        result_text = '\n'.join(
            f'Авто {i+1}: RGB = {color}' for i, color in enumerate(rgb_colors)
        )

        return result_text, vis_image
    except Exception as e:
        return f'Ошибка обработки: {str(e)}', None


# Создание интерфейса Gradio
with gr.Blocks() as interface:
    gr.Markdown('# Определение цвета автомобилей')
    gr.Markdown('Загрузите изображение, чтобы определить цвета автомобилей.')
    with gr.Row():
        input_image = gr.Image(type='pil',
                               label='Загрузите фотографию автомобиля')
    with gr.Row():
        output_text = gr.Textbox(label='Цвета автомобилей (RGB)')
        output_image = gr.Image(label='Результат обработки')
    input_image.change(fn=demo_process_image,
                       inputs=input_image,
                       outputs=[output_text, output_image])

if __name__ == '__main__':
    interface.launch(server_name="0.0.0.0", server_port=7860)
