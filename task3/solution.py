import cv2
import pytesseract


# Указываем перечень контрольных (уникальных) слов для определения типа.
DOCUMENTS_TYPE = {
    'договор': ['договор',],
    'паспорт': ['отделом', 'мвд'],
    'СТС': ['certificat'],
    'ИНН': ['налогам'],
    'права': ['водительское'],
}


# Само распознавание.
def ocr(file_path):
    """Принимаем путь к картинке и возвращаем сырые данные."""
    image = cv2.imread(file_path)
    # Конвертация изображения в градации серого.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Конфиг для лучшего распознавания.
    custom_config = r'--oem 3 --psm 6'
    # Распознаем.
    text = pytesseract.image_to_string(
        gray_image,
        lang='rus+eng',
        config=custom_config
    )
    return set(text.lower().split())         # множество для скорости работы.


def define_type(raw_text):
    for key, values in DOCUMENTS_TYPE.items():
        for value in values:
            if value in raw_text:
                return key
    return 'Не удалось распознать документ.'


if __name__ == '__main__':
    image_path = 'task3/data/$2y$10$k1fd1.d6HmhGzjzlTay.ChRiDY8LgriFg.EupH6kUCTCt9fjRm.png'
    raw_text = ocr(image_path)
    result = define_type(raw_text)
    print(result)
