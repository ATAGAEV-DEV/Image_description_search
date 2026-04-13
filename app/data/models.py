import os

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA = "pictures"


def get_engine(schema: str) -> AsyncEngine:
    """Создаёт и возвращает асинхронный движок SQLAlchemy с указанным схемой.

    Устанавливает параметр search_path в соединении, чтобы все запросы выполнялись
    в заданной схеме PostgreSQL.
    """
    return create_async_engine(
        DATABASE_URL,
        connect_args={"server_settings": {"search_path": schema}},
        pool_pre_ping=True,
        pool_recycle=1800,
    )


if SCHEMA is None or SCHEMA == "":
    engine = get_engine("public")
else:
    engine = get_engine(SCHEMA)
async_session = async_sessionmaker(engine)


class Base(AsyncAttrs, DeclarativeBase):
    """Базовый класс для всех моделей, поддерживающий асинхронные атрибуты."""

    pass


class ImageDescription(Base):
    """Модель таблицы 'image_descriptions' для хранения текстовых описаний."""

    __tablename__ = "image_descriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())


class ProcessedImageDescriptions(Base):
    """Модель таблицы 'processed_image_descriptions' для отслеживания обработанных фото."""

    __tablename__ = "processed_image_descriptions"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())


class Users(Base):
    """Модель таблицы 'users' для хранения информации о пользователях.

    Также нужен для проверки доступа пользователя к боту.
    """

    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=False)
    username = Column(String)


async def init_models() -> None:
    """Создает таблицы в базе данных, если они не существуют.

    Также синхронизирует sequence таблицы image_descriptions с max(id),
    чтобы autoincrement корректно работал после вставки данных с явными ID.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from app.data.request import reset_image_description_sequence

    await reset_image_description_sequence()
