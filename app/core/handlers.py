import os

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, Message

import app.data.request as req
import app.tools.utils as ut
from app.core.image_to_text import ai_generate, generate_prompt
from app.services.llama_integration import LlamaIndexManager

router = Router()

ACCESS_PASSWORD = "e5ae93bd8095fbd86c25a110bbf194a5a1a209f1e8eb31bb30c8b0ecbe254d58"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PICTURES_DIR = os.path.join(".", "app", "pictures")


class RegisterState(StatesGroup):
    """Состояния конечного автомата (FSM) для процесса регистрации пользователя."""

    waiting_for_password = State()


@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    """Обрабатывает команду /start.

    Проверяет, зарегистрирован ли пользователь в системе.
    Если пользователь найден, приветствует и предлагает начать поиск.
    Если пользователь не найден, запрашивает ввод пароля для доступа и
    переводит в состояние ожидания пароля.
    """
    user_id = message.from_user.id
    user = await req.get_user_by_id(user_id)
    if user:
        await message.answer("Добро пожаловать!")
    else:
        await message.answer("Добро пожаловать! Для продолжения работы введите пароль для доступа.")
        await state.set_state(RegisterState.waiting_for_password)


@router.message(RegisterState.waiting_for_password)
async def password_handler(message: Message, state: FSMContext) -> None:
    """Обрабатывает ввод пароля пользователем для авторизации.

    Проверяет введённый пароль, сравнивая его хеш с заданным ACCESS_PASSWORD.
    При успешной авторизации добавляет пользователя в базу и очищает состояние FSM.
    В случае неверного пароля отправляет сообщение с просьбой повторить ввод.
    """
    user_id = message.from_user.id
    if ut.hash_password(message.text.strip()) == ACCESS_PASSWORD:
        await req.add_user(user_id, message.from_user.username or "")
        await message.answer("Авторизация успешна! Теперь у вас полный доступ.")
        await state.clear()
    else:
        await message.answer("Неверный пароль. Попробуйте еще раз:")


@router.message(Command("sync"))
async def sync_images_handler(message: Message):
    """Синхронизирует изображения из БД в ChromaDB с защитой от дубликатов"""
    user_id = message.from_user.id
    user = await req.get_user_by_id(user_id)
    if not user:
        await message.answer("Сначала авторизуйтесь с помощью /start")
        return

    await message.answer("🔄 Начинаю синхронизацию...")

    try:
        all_descriptions = await req.get_all_image_descriptions()
        processed_ids = await req.get_processed_image_ids()

        new_descriptions = [desc for desc in all_descriptions if desc.id not in processed_ids]

        if not new_descriptions:
            await message.answer("✅ Нет новых данных для синхронизации")
            return

        await message.answer(f"📥 Найдено {len(new_descriptions)} новых записей")

        images_data = []
        for desc in new_descriptions:
            images_data.append({"id": desc.id, "name": desc.name, "description": desc.description})

        llama_manager = LlamaIndexManager()
        success = await llama_manager.index_images(images_data)

        if success:
            for desc in new_descriptions:
                await req.add_processed_image_description(desc)

            await message.answer(
                f"✅ Синхронизация завершена!\n"
                f"Добавлено: {len(new_descriptions)} записей\n"
                f"Всего в базе: {len(all_descriptions)} записей"
            )
        else:
            await message.answer("❌ Ошибка при индексации в ChromaDB")

    except Exception as e:
        await message.answer(f"❌ Ошибка синхронизации: {str(e)}")


@router.message(Command("stats"))
async def stats_handler(message: Message):
    """Показывает статистику базы данных"""
    user_id = message.from_user.id
    user = await req.get_user_by_id(user_id)
    if not user:
        await message.answer("Сначала авторизуйтесь с помощью /start")
        return

    try:
        all_descriptions = await req.get_all_image_descriptions()
        processed_ids = await req.get_processed_image_ids()

        llama_manager = LlamaIndexManager()
        chroma_stats = await llama_manager.get_collection_stats()

        stats_text = (
            "📊 Статистика базы данных:\n"
            f"• Всего записей в БД: {len(all_descriptions)}\n"
            f"• Обработано записей: {len(processed_ids)}\n"
            f"• Новых для обработки: {len(all_descriptions) - len(processed_ids)}\n"
            f"• В ChromaDB: {chroma_stats.get('documents_count', 'N/A')}\n"
        )

        await message.answer(stats_text)

    except Exception as e:
        await message.answer(f"❌ Ошибка получения статистики: {str(e)}")


@router.message(Command("search"))
async def search_images_handler(message: Message):
    """Ищет изображения по описанию и отправляет найденные файлы"""
    user_id = message.from_user.id
    user = await req.get_user_by_id(user_id)
    if not user:
        await message.answer("Сначала авторизуйтесь с помощью /start")
        return

    search_query = message.text.replace("/search", "").strip()

    if not search_query:
        await message.answer("Введите запрос для поиска:\n/search <ваш запрос>")
        return

    await message.answer(f'🔍 Ищу: "{search_query}"')

    try:
        llama_manager = LlamaIndexManager()
        results = await llama_manager.search_images(search_query, limit=5)

        if not results:
            await message.answer("😔 Ничего не найдено")
            return

        for result in results:
            image_name = result["name"]
            image_path = os.path.join(".", "app", "pictures", image_name)

            if os.path.exists(image_path):
                photo = FSInputFile(image_path)
                await message.answer_photo(photo, caption=f"📸 {image_name}")
            else:
                await message.answer(f"❌ Файл {image_name} не найден в папке pictures")

    except Exception as e:
        await message.answer(f"❌ Ошибка поиска: {str(e)}")


@router.message(F.photo)
async def photo_upload_handler(message: Message):
    """Сохраняет фотографии, отправленные пользователями, в папку pictures.

    Проверяет расширение файла, сохраняет изображение, затем обрабатывает его
    с помощью AI для получения описания и сохранения в базу данных.
    """
    user_id = message.from_user.id
    user = await req.get_user_by_id(user_id)
    if not user:
        await message.answer("Сначала авторизуйтесь с помощью /start")
        return

    if not message.photo:
        await message.answer("❌ Не удалось получить изображение")
        return

    photo = message.photo[-1]

    try:
        file = await message.bot.get_file(photo.file_id)

        if not file.file_path:
            await message.answer("❌ Не удалось получить путь к файлу")
            return

        if file.file_path:
            _, extension = os.path.splitext(file.file_path)
        else:
            extension = ""

        if not extension or extension.lower() not in ALLOWED_EXTENSIONS:
            await message.answer(
                f"❌ Недопустимое расширение файла. Поддерживаемые форматы: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            return

        os.makedirs(PICTURES_DIR, exist_ok=True)

        filename = f"{photo.file_unique_id}{extension}"
        destination = os.path.join(PICTURES_DIR, filename)

        if os.path.exists(destination):
            await message.answer(
                f"❌ Файл с таким именем уже существует: {os.path.basename(destination)}. "
                f"Пожалуйста, отправьте другое изображение."
            )
            return

        await message.bot.download(photo.file_id, destination=str(destination))
        await message.answer(f"📸 Изображение сохранено как {os.path.basename(destination)}")

        await message.answer("🤖 Обрабатываю изображение...")

        prompt = generate_prompt(destination)
        description = await ai_generate(prompt)

        if not description:
            print(f"Не удалось получить описание для {filename}, пропуск файла\n")
            if os.path.exists(destination):
                os.remove(destination)
            await message.answer("❌ Не удалось обработать изображение")
            return

        success = await req.create_image_description(os.path.basename(destination), description)

        if success:
            await message.answer(
                f"✅ Изображение успешно обработано!\n\n📝 Описание:\n{description}"
            )
        else:
            if os.path.exists(destination):
                os.remove(destination)
            await message.answer("❌ Не удалось сохранить описание изображения в базу данных")

    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке изображения: {str(e)}")
        print(f"Ошибка в photo_upload_handler: {str(e)}")
