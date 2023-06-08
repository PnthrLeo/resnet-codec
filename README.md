# Кодек на ResNet18

## Запуск
1. Скачать веса модели по [ссылке](https://disk.yandex.ru/d/uZvhVuvxmsOj4g).

2. Установить зависимости:
`pip install -r requirements.txt`

3. Запустить скрипт (Quality - число от 1 до 6):
`python coder.py --model_path <path_to_model> --image_path <path_to_image> --bin_path <path_to_bin> --quality <quality>`

4. Для декодирования запустить скрипт:
`python decoder.py --model_path <path_to_model> --bin_path <path_to_bin> --image_path <path_to_image>`
