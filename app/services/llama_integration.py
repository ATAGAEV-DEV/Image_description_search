import asyncio
import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import AsyncOpenAI

load_dotenv()

AI_TOKEN_PROXYAPI = os.getenv("AI_TOKEN")
AI_TOKEN_AITUNNEL = os.getenv("AI_TOKEN1")
AI_TOKEN_POLZA = os.getenv("AI_TOKEN_POLZA")
proxyapi = "https://api.proxyapi.ru/openai/v1"
aitunnel = "https://api.aitunnel.ru/v1/"
polza = "https://api.polza.ai/api/v1"


class LlamaIndexManager:
    """Управляет интеграцией с LlamaIndex и ChromaDB для работы с описаниями изображений."""

    def __init__(self, collection_name: str = "image_descriptions") -> None:
        """Инициализирует менеджер LlamaIndex и ChromaDB."""
        self.custom_client = AsyncOpenAI(
            api_key=AI_TOKEN_AITUNNEL,
            base_url=aitunnel,
        )

        self.embed_model = OpenAIEmbedding(
            api_key=AI_TOKEN_AITUNNEL,
            api_base=aitunnel,
            model="text-embedding-3-large",
        )

        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=32)
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name

        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser

    def get_collection(self) -> Any:
        """Получить или создать коллекцию для изображений."""
        try:
            collection = self.db.get_collection(self.collection_name)
        except Exception:
            collection = self.db.create_collection(self.collection_name)
        return collection

    async def index_images(self, images: list[dict[str, Any]]) -> bool:
        """Индексирует описания картинок в ChromaDB."""
        try:
            documents = []
            for img in images:
                document = Document(
                    text=img["description"],
                    metadata={
                        "document_type": "image_description",
                        "image_id": img["id"],
                        "name": img["name"],
                    },
                    id_=str(img["id"]),
                )
                documents.append(document)

            if not documents:
                return False

            collection = self.get_collection()
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            await asyncio.to_thread(
                VectorStoreIndex.from_documents,
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
            return True

        except Exception as e:
            print(f"Ошибка индексации изображений: {e}")
            return False

    async def search_images(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Ищет изображения по текстовому запросу (только по описаниям)."""
        try:
            collection = self.get_collection()
            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = await asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store)

            retriever = index.as_retriever(similarity_top_k=limit)
            nodes = await asyncio.to_thread(retriever.retrieve, query)

            results = []
            for node in nodes:
                results.append(
                    {
                        "image_id": node.metadata.get("image_id"),
                        "name": node.metadata.get("name"),
                        "description": node.text,
                    }
                )

            return results

        except Exception as e:
            print(f"Ошибка поиска изображений: {e}")
            return []

    async def clear_collection(self) -> bool:
        """Очищает коллекцию изображений."""
        try:
            self.db.delete_collection(self.collection_name)
            self.get_collection()
            return True
        except Exception as e:
            print(f"Ошибка очистки коллекции: {e}")
            return False

    async def get_collection_stats(self) -> dict[str, Any]:
        """Возвращает статистику коллекции."""
        try:
            collection = self.get_collection()
            count = collection.count()
            return {"collection_name": self.collection_name, "documents_count": count}
        except Exception as e:
            print(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
