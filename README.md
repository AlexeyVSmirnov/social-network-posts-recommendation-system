# Рекомендательная система сообщений в соц. сети

В этом проекте представлено решение по созданию рекомендательной системы сообщений в некоторой соц. сети и веб-сервиса для получения рекомендаций для определенного пользователя.

## Стек
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Uvicorn](https://img.shields.io/badge/uvicorn-9bdbf0?style=for-the-badge)
![Catboost](https://img.shields.io/badge/catboost-ffcd1f?style=for-the-badge)
![Gensim](https://img.shields.io/badge/gensim-5325b3?style=for-the-badge)
![SQLAlchemy](https://img.shields.io/badge/SQL_Alchemy-cd2103?style=for-the-badge)
![SQL](https://img.shields.io/badge/SQL-white?logo=SQL&s&style=for-the-badge)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)

## Цель
Создать систему рекомендаций и веб-сервис, который по идентификатору пользователя соц. сети на заданный момент времени возвращал бы список постов, которые с большой вероятностью представляют интерес для выбранного пользователя.
## Исходные данные
Для решения поставленной задачи используются исторические данные соц. сети из базы данных PostgreSQL. Параметры подключения к базе данных должны быть указаны в файле `./creds/.env`. Пример заполнения этого файла можно посмотреть в [./creds/.env.example](./creds/.env.example).  
На случай, если у пользователя нет требуемых параметров для подключения к базе данных, есть возможность запустить сервис в демонстрационном режиме, в котором необходимые для работы данные будут браться из файлов в директории `./data/`.
Переключение между режимами выполняется в конфигурационном файле [./config.json](./config.json) параметром "db"."use_db".
## Этапы создания сервиса
- Предобработка и сохранение данных о пользователях и сообщениях в соц. сети, выделение значимых признаков. (см. [./notebooks/prepare_and_save_features.ipynb](./notebooks/prepare_and_save_features.ipynb) и [./notebooks/get_extended_posts_features.ipynb](./notebooks/get_extended_posts_features.ipynb))  
- Обучение и сохранение модели CatBoostClassifier для прогнозирования реакции пользователей на сообщения в соц. сети (см. [./notebooks/model_train_and_save.ipynb](./notebooks/model_train_and_save.ipynb) и [./notebooks/get_extended_posts_features.ipynb](./notebooks/get_extended_posts_features.ipynb))  
- Создание сервиса рекомендаций с использованием фреймворка FastAPI, использующего обученную модель и признаки для получения рекомендаций для конкретного пользователя.
## Установка и запуск
Для запуска сервиса клонировать репозиторий следующей командой:  
```
git clone https://github.com/AlexeyVSmirnov/social-network-posts-recommendation-system.git
```
Далее в директории репозитория создать и активировать виртуальное окружение и установить требуемые зависимости следующей командой:  
```
pip install -r requirements.txt
```
Для запуска сервиса нужно выполнить команду
```
python.exe -m uvicorn app:app
```
После завершения процесса запуска сервис доступен по адресу [http://127.0.0.1:8000/post/recommendations/](http://127.0.0.1:8000/post/recommendations/)  
Примеры запросов к сервису:
```
http://127.0.0.1:8000/post/recommendations/?id=204&time=2021-11-29&limit=3

http://127.0.0.1:8000/post/recommendations/?id=265&time=2022-06-01&limit=1

http://127.0.0.1:8000/post/recommendations/?id=9876&time=2023-10-15&limit=10
```
## Результаты
В этом проекте был создан сервис рекомендаций пользователям сообщений соц.сети предположительно интересных для пользователей. Для проверки качества рекомендаций была использована метрика Hitrate@5.  
Кроме того, в рамках проекта было создано две модели рекомендаций, контрольная и тестовая, и создан механизм разбиения пользователей на две группы, также контрольную и тестовую. Это сделано для обеспечения возможности проведения A/B-тестирования для сравнения двух моделей.  
Итоговые метрики по каждой из моделей:
- Среднее значение Hitrate@5 для контрольной модели: 0.564
- Среднее значение Hitrate@5 для тестовой модели: 0.572