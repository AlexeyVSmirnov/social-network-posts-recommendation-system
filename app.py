"""Сервис получения рекомендованных сообщений в соцсети для конкретного пользователя.
Сервис может работать в двух режимах — рабочем и демонстрационном. 

Рабочий режим включается в конфигурационном файле config.json установкой db.use_db = true. 
В этом режиме сервис берет данные из базы данных, и для корректной работы требуются 
настройки подключения к базе данных в скрытом файле './creds/.env'.

При установке db.use_db = false сервис работает в демонстрационном режиме, 
и все необходимые данные берутся из заранее сохраненных файлов в директории './data/'.
"""

import hashlib
import json
from datetime import date
import pandas as pd
import numpy as np
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from catboost import CatBoostClassifier
from loguru import logger
from dotenv import dotenv_values
from schema import Response


DOTENV_PATH = "./creds/.env"


def get_config_data():
    """Возвращает словарь с настройками сервиса из файла config.json"""

    with open("./config.json", "r") as cfg:
        data = json.load(cfg)
    return data


def get_connection_string():
    """Возвращает строку подключения к базе PostgreSQL
    по данным в скрытом файле настроек .env.
    """

    env_dict = dotenv_values(DOTENV_PATH)

    server = env_dict.get("KC_STARTML_POSTGRES_SERVER", "")
    port = env_dict.get("KC_STARTML_POSTGRES_PORT")
    db_name = env_dict.get("KC_STARTML_POSTGRES_DB")
    user = env_dict.get("KC_STARTML_POSTGRES_USER", "")
    password = env_dict.get("KC_STARTML_POSTGRES_PASSWORD")

    password = (":" + password) if password else ""
    port = (":" + port) if port else ""
    db_name = ("/" + db_name) if db_name else ""

    return f"postgresql://{user}{password}@{server}{port}{db_name}"


def get_feature_tables_names():
    """Возвращает словарь с именами таблиц из базы PostgreSQL, 
    в которых содержатся ранее рассчитанные признаки пользователей 
    и сообщений в соцсети.
    """

    env_dict = dotenv_values(DOTENV_PATH)
    return {
        "users_features_table": env_dict.get("KC_STARTML_USERS_FEATURES_TABLE", ""),
        "posts_features_table_control": env_dict.get("KC_STARTML_POSTS_FEATURES_TABLE", ""),
        "posts_features_table_test": env_dict.get("KC_STARTML_POSTS_FEATURES_EXT_TABLE", ""),
        "posts_texts_table": env_dict.get("KC_STARTML_POSTS_TEXTS_TABLE", ""),
        "feed_data_table": env_dict.get("KC_STARTML_FEED_DATA_TABLE", "")
    }


config_data = get_config_data()

SALT = config_data["test_group"]["split_salt"]
TEST_PERCENT = config_data["test_group"]["split_percent"]
TEST_GROUP_NAME = config_data["test_group"]["group_name"]
CONTROL_GROUP_NAME = config_data["control_group"]["group_name"]

MODEL_FILENAME_CONTROL = config_data["control_group"]["model_file_name"]
MODEL_FILENAME_TEST = config_data["test_group"]["model_file_name"]

MODEL_COLUMNS_CONTROL = config_data["control_group"]["model_columns"]
MODEL_COLUMNS_TEST = config_data["test_group"]["model_columns"]

USERS_FEATURES_LOCAL_FILE = config_data["local_data"]["user_features"]
POSTS_FEATURES_LOCAL_FILE_CONTROL = config_data["local_data"]["post_features_control"]
POSTS_FEATURES_LOCAL_FILE_TEST = config_data["local_data"]["post_features_test"]
POSTS_TEXTS_LOCAL_FILE = config_data["local_data"]["post_texts"]

USE_DB = config_data["db"]["use_db"]
BATCH_LOAD_CHUNKSIZE = config_data["db"]["batch_load_chunk_size"]

feature_tables_names = get_feature_tables_names()
USERS_FEATURES_TABLE = feature_tables_names["users_features_table"]
POSTS_FEATURES_TABLE_CONTROL = feature_tables_names["posts_features_table_control"]
POSTS_FEATURES_TABLE_TEST = feature_tables_names["posts_features_table_test"]
POSTS_TEXTS_TABLE = feature_tables_names["posts_texts_table"]
FEED_DATA_TABLE = feature_tables_names["feed_data_table"]

SQLALCHEMY_DATABASE_URL = ""
if USE_DB:
    SQLALCHEMY_DATABASE_URL = get_connection_string()
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_exp_group(user_id: int) -> str:
    """Функции разбиения пользователей на две группы

    Parameters
    ----------
    user_id : int
        Идентификатор пользователя

    Returns
    -------
    str
        Код группы пользователя

    """
    is_test = int(hashlib.md5(f"{user_id}{SALT}".encode()).hexdigest(), 16) % 100 <= TEST_PERCENT
    return TEST_GROUP_NAME if is_test else CONTROL_GROUP_NAME


def get_json_from_df(df_data: pd.DataFrame):
    """Возвращает json с данными из переданного DataFrame."""
    res = df_data.to_json(orient="records")
    parsed = json.loads(res)
    return parsed


def load_models_control():
    """Загружает контрольную модель catboost из файла и возвращает её."""
    model = CatBoostClassifier()
    model.load_model(MODEL_FILENAME_CONTROL)
    return model


def load_models_test():
    """Загружает тестовую модель catboost из файла и возвращает её."""
    model = CatBoostClassifier()
    model.load_model(MODEL_FILENAME_TEST)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    """Выполняет пакетную загрузку данных из базы данных 
    по переданному запросу.

    Parameters
    ----------
    query : str
        Строка запроса в базу данных

    Returns
    -------
    pandas.DataFrame
        Результат запроса из базы данных

    """
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=BATCH_LOAD_CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_users_features() -> pd.DataFrame:
    """Загружает признаки пользователей и возвращает их. 
    Если USE_DB установлено в True, то данные загружаются 
    из базы данных. В противном случае данные загружаются 
    из указанного в настройках файла.
    """
    if USE_DB:
        result = batch_load_sql(f"SELECT * FROM {USERS_FEATURES_TABLE}")
    else:
        result = pd.read_csv(USERS_FEATURES_LOCAL_FILE)
    return result


def load_posts_features_control() -> pd.DataFrame:
    """Загружает признаки сообщений соцсети для контрольной 
    модели и возвращает их. Если в USE_DB установлено True, 
    то данные загружаются из базы данных. В противном случае 
    данные загружаются из указанного в настройках файла.
    """
    if USE_DB:
        result = batch_load_sql(f"SELECT * FROM {POSTS_FEATURES_TABLE_CONTROL}")
    else:
        result = pd.read_csv(POSTS_FEATURES_LOCAL_FILE_CONTROL)
    return result


def load_posts_features_test() -> pd.DataFrame:
    """Загружает признаки сообщений соцсети для тестовой 
    модели и возвращает их. Если в USE_DB установлено True, 
    то данные загружаются из базы данных. В противном случае 
    данные загружаются из указанного в настройках файла.
    """
    if USE_DB:
        result = batch_load_sql(f"SELECT * FROM {POSTS_FEATURES_TABLE_TEST}")
    else:
        result = pd.read_csv(POSTS_FEATURES_LOCAL_FILE_TEST)
    return result


def load_posts_data() -> pd.DataFrame:
    """Загружает данные сообщений соцсети и возвращает их. 
    Если в USE_DB установлено True, то данные загружаются 
    из базы данных. В противном случае данные загружаются 
    из указанного в настройках файла.
    """
    if USE_DB:
        result = pd.read_sql(f"SELECT post_id, text, topic FROM {POSTS_TEXTS_TABLE}",
                        con=SQLALCHEMY_DATABASE_URL)
    else:
        result = pd.read_csv(POSTS_TEXTS_LOCAL_FILE)
    return result


# Функция загрузки лайков по датам
def load_likes_data() -> pd.DataFrame:
    """Загружает данные о 'лайках' пользователей. 
    Если в USE_DB установлено True, то данные загружаются 
    из базы данных. В противном случае возвращается пустой
    DataFrame.
    """
    if USE_DB:
        result = batch_load_sql("SELECT timestamp, user_id, post_id "
                            +f"FROM {FEED_DATA_TABLE} WHERE action = 'like' ")
    else:
        result = pd.DataFrame(columns=["timestamp", "user_id", "post_id"])
    return result


# Загрузка необходимых для работы данных
logger.info("Control model loading...")
m_model_control = load_models_control()
logger.info("Control model loaded")

logger.info("Test model loading...")
m_model_test = load_models_test()
logger.info("Test model loaded")

logger.info("Users features loading...")
m_df_users_features = load_users_features()
logger.info("Users features loaded")

logger.info("Posts_features for control model loading...")
m_df_posts_features_control = load_posts_features_control()
logger.info("Posts_features for control model loaded")

logger.info("Posts_features for test model loading...")
m_df_posts_features_test = load_posts_features_test()
logger.info("Posts_features for test model loaded")

logger.info("Posts data loading...")
m_df_posts_data = load_posts_data()
logger.info("Posts data loaded")

logger.info("Likes data loading...")
m_df_likes_data = load_likes_data()
logger.info("Likes data loaded")

app = FastAPI()

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
		id: int,
		time: date,
		limit: int = 10) -> Response:
    """"Эндпоинт сервиса рекомендаций сообщений в соцсети.

    Parameters
    ----------
    id : int
        Идентификатор пользователя
    time : date
        Дата, на которую дается рекомендация
    limit : int
        Количество рекомендуемых сообщений.

    Returns
    -------
    Response
        Объект, содержащий группу пользователя и список
        рекомендуемых ему сообщений.

    
    """
    exp_group = get_exp_group(id)

    if exp_group == TEST_GROUP_NAME:
        model_columns = MODEL_COLUMNS_TEST
        m_model = m_model_test
        m_df_posts_features = m_df_posts_features_test
    elif exp_group == CONTROL_GROUP_NAME:
        model_columns = MODEL_COLUMNS_CONTROL
        m_model = m_model_control
        m_df_posts_features = m_df_posts_features_control
    else:
        return {}

    time = np.datetime64(time)
    df_user_likes = m_df_likes_data[(m_df_likes_data["user_id"]==id)
                                    & (m_df_likes_data["timestamp"]<=time)]
    df_preds = (m_df_users_features[m_df_users_features["user_id"]==id]
                .merge(m_df_posts_features, how="cross"))
    df_preds = df_preds[~df_preds["post_id"].isin(df_user_likes["post_id"])]
    df_preds["proba"] = m_model.predict_proba(df_preds[model_columns])[:, 1]
    df_result = (df_preds.sort_values("proba", ascending=False)[:limit][["post_id"]]
                 .merge(m_df_posts_data, on="post_id"))
    df_result = df_result.rename(columns={"post_id":"id"})
    return  {"exp_group": exp_group, "recommendations": get_json_from_df(df_result)}
