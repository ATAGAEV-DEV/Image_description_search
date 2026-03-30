import asyncio

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.data.models import (
    ImageDescription,
    ProcessedImageDescriptions,
    Users,
    async_session,
)

DB_TIMEOUT = 10


async def get_user_by_id(user_id: int) -> Users | None:
    """Получает пользователя из базы данных по его идентификатору.

    Выполняет асинхронный запрос к базе данных для поиска записи в таблице 'users'
    с указанным user_id. Возвращает объект пользователя или None, если пользователь
    не найден. В случае ошибки выводит сообщение об ошибке и возвращает None.
    """
    try:
        async with async_session() as session:
            query = select(Users).where(Users.user_id == user_id)
            result = await asyncio.wait_for(session.execute(query), timeout=DB_TIMEOUT)
            return result.scalar_one_or_none()
    except TimeoutError:
        print(f"Таймаут при получении пользователя с user_id={user_id}")
        return None
    except Exception as e:
        print(f"Ошибка получения пользователя: {e}")
        return None


async def add_user(user_id: int, username: str) -> None:
    """Добавляет нового пользователя или обновляет существующего в базе данных.

    Использует upsert (INSERT ... ON CONFLICT DO UPDATE) для безопасного
    добавления пользователя. Если пользователь уже существует, обновляется username.
    """
    try:
        async with async_session() as session:
            stmt = pg_insert(Users).values(user_id=user_id, username=username)
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id"],
                set_={"username": username},
            )
            await asyncio.wait_for(session.execute(stmt), timeout=DB_TIMEOUT)
            await asyncio.wait_for(session.commit(), timeout=DB_TIMEOUT)
    except TimeoutError:
        print(f"Таймаут при добавлении пользователя с user_id={user_id}")
    except Exception as e:
        print(f"Ошибка добавления пользователя: {e}")


async def get_all_image_descriptions() -> list[ImageDescription]:
    """Получает все записи из ImageDescription."""
    try:
        async with async_session() as session:
            query = select(ImageDescription)
            result = await asyncio.wait_for(session.execute(query), timeout=60)
            return result.scalars().all()
    except TimeoutError:
        print("Таймаут при получении всех описаний")
        return []
    except Exception as e:
        print(f"Ошибка получения всех описаний: {e}")
        return []


async def get_processed_image_ids() -> set[int]:
    """Получает ID уже обработанных записей."""
    try:
        async with async_session() as session:
            query = select(ProcessedImageDescriptions.id)
            result = await asyncio.wait_for(session.execute(query), timeout=60)
            return set([row[0] for row in result])
    except TimeoutError:
        print("Таймаут при получении обработанных ID")
        return set()
    except Exception as e:
        print(f"Ошибка получения обработанных ID: {e}")
        return set()


async def add_processed_image_description(image_desc: ImageDescription) -> None:
    """Добавляет запись в ProcessedImageDescriptions."""
    try:
        async with async_session() as session:
            processed_record = ProcessedImageDescriptions(
                id=image_desc.id,
                name=image_desc.name,
                description=image_desc.description,
            )
            session.add(processed_record)
            await asyncio.wait_for(session.commit(), timeout=DB_TIMEOUT)
    except TimeoutError:
        print(f"Таймаут при добавлении описания {image_desc.id}")
    except Exception as e:
        print(f"Ошибка добавления в обработанные: {e}")


async def reset_image_description_sequence() -> None:
    """Синхронизирует PostgreSQL sequence таблицы image_descriptions с max(id).

    Необходимо вызывать при старте бота, чтобы autoincrement генерировал
    корректные ID после вставки данных с явными ID.
    """
    try:
        async with async_session() as session:
            await session.execute(
                text(
                    "SELECT setval("
                    "pg_get_serial_sequence('image_descriptions', 'id'), "
                    "COALESCE((SELECT MAX(id) FROM image_descriptions), 0))"
                )
            )
            await session.commit()
            print("Sequence image_descriptions синхронизирован")
    except Exception as e:
        print(f"Ошибка синхронизации sequence: {e}")


async def create_image_description(name: str, description: str, max_retries: int = 3) -> bool:
    """Вставляет новую запись в таблицу image_descriptions.

    Использует retry-логику с экспоненциальной задержкой для устойчивости
    к transient-ошибкам БД.
    """
    for attempt in range(max_retries):
        try:
            async with async_session() as session:
                new_record = ImageDescription(name=name, description=description)
                session.add(new_record)
                await asyncio.wait_for(session.commit(), timeout=DB_TIMEOUT)
                return True

        except TimeoutError:
            print(
                f"Таймаут при сохранении описания для {name} (попытка {attempt + 1}/{max_retries})"
            )

        except SQLAlchemyError as e:
            print(
                f"Ошибка БД при сохранении описания для {name} "
                f"(попытка {attempt + 1}/{max_retries}): {e}"
            )

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5 * (2**attempt))

    print(f"Не удалось сохранить описание для {name} после {max_retries} попыток")
    return False
