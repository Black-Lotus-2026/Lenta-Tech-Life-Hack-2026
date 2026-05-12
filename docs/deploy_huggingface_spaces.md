# Деплой на Hugging Face Spaces

1. Создайте Space: SDK = Docker, Visibility = Public.
2. Загрузите файлы репозитория.
3. Для максимального качества добавьте обученные веса в `models/price_tag_yolo.pt`.
4. Если хватает RAM/CPU, в Dockerfile раскомментируйте `pip install -r requirements-full.txt`.
5. Space автоматически запустит `python app.py` на порту 7860.

Для Render/Fly/Koyeb используется тот же Dockerfile. Команда запуска: `python app.py`.
