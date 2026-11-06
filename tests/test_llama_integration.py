"""Real LlamaIndexManager logic with offline service boundaries and real nodes."""

import copy
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch, sentinel

from llama_index.core.schema import NodeWithScore, TextNode

from tests.support import TEST_ENVIRONMENT, import_application

llama_integration = import_application("app.services.llama_integration")


class _LlamaBoundariesMixin(unittest.TestCase):
    def patch_attribute(self, target, attribute: str, **kwargs):
        patcher = patch.object(target, attribute, **kwargs)
        replacement = patcher.start()
        self.addCleanup(patcher.stop)
        return replacement

    def setUp(self) -> None:
        super().setUp()
        for target, attribute in (
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
        ):
            guard = self.patch_attribute(
                target, attribute, side_effect=AssertionError("Network access is forbidden")
            )
            self.addCleanup(guard.assert_not_called)

        self.embedding_factory = self.patch_attribute(llama_integration, "OpenAIEmbedding")
        self.embedding_factory.return_value = sentinel.embedding
        # The parser is a local boundary too: its default tokenizer may download data.
        self.parser_factory = self.patch_attribute(llama_integration, "SimpleNodeParser")
        self.parser_factory.from_defaults.return_value = sentinel.node_parser
        self.collection = Mock(name="test_collection", spec=["count"])
        self.collection.count.return_value = 3
        self.db = Mock(
            name="test_database", spec=["get_collection", "create_collection", "delete_collection"]
        )
        self.db.get_collection.return_value = self.collection
        self.db.create_collection.return_value = self.collection
        self.db.delete_collection.return_value = None
        self.db_factory = self.patch_attribute(
            llama_integration.chromadb, "PersistentClient", return_value=self.db
        )
        self.settings = SimpleNamespace()
        self.patch_attribute(llama_integration, "Settings", new=self.settings)
        self.vector_store_factory = self.patch_attribute(llama_integration, "ChromaVectorStore")
        self.vector_store_factory.return_value = sentinel.vector_store
        self.storage_context_factory = self.patch_attribute(llama_integration, "StorageContext")
        self.storage_context_factory.from_defaults.return_value = sentinel.storage_context
        self.index_factory = self.patch_attribute(llama_integration, "VectorStoreIndex")
        self.index = Mock(name="index", spec=["as_retriever"])
        self.retriever = Mock(name="retriever", spec=["retrieve"])
        self.index.as_retriever.return_value = self.retriever
        self.retriever.retrieve.return_value = []
        self.index_factory.from_vector_store.return_value = self.index
        self.index_factory.from_documents.return_value = sentinel.built_index


class LlamaIndexManagerInitializationTests(_LlamaBoundariesMixin, unittest.TestCase):
    def test_default_initialization_configures_embedding_parser_database_and_local_settings(
        self,
    ) -> None:
        manager = llama_integration.LlamaIndexManager()

        self.embedding_factory.assert_called_once_with(
            api_key=TEST_ENVIRONMENT["AI_TOKEN"],
            api_base=TEST_ENVIRONMENT["AI_PROVIDER_URL"],
            model="text-embedding-3-large",
        )
        self.parser_factory.from_defaults.assert_called_once_with(chunk_size=512, chunk_overlap=32)
        self.db_factory.assert_called_once_with(path="./chroma_db")
        self.assertIs(manager.embed_model, sentinel.embedding)
        self.assertIs(manager.node_parser, sentinel.node_parser)
        self.assertIs(manager.db, self.db)
        self.assertEqual(manager.collection_name, "image_descriptions")
        self.assertIs(self.settings.embed_model, manager.embed_model)
        self.assertIs(self.settings.node_parser, manager.node_parser)
        self.db.get_collection.assert_not_called()
        self.db.create_collection.assert_not_called()

    def test_custom_collection_name_is_preserved_without_creating_a_collection(self) -> None:
        manager = llama_integration.LlamaIndexManager(collection_name="unit-test-images")

        self.assertEqual(manager.collection_name, "unit-test-images")
        self.db.get_collection.assert_not_called()
        self.db.create_collection.assert_not_called()
        self.db_factory.assert_called_once_with(path="./chroma_db")

    def test_embedding_initialization_failure_propagates_before_database_access(self) -> None:
        error = ValueError("invalid test embedding configuration")
        self.embedding_factory.side_effect = error

        with self.assertRaises(ValueError) as caught:
            llama_integration.LlamaIndexManager()

        self.assertIs(caught.exception, error)
        self.parser_factory.from_defaults.assert_not_called()
        self.db_factory.assert_not_called()
        self.assertEqual(vars(self.settings), {})

    def test_parser_initialization_failure_propagates_before_database_access(self) -> None:
        error = RuntimeError("parser setup failed")
        self.parser_factory.from_defaults.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            llama_integration.LlamaIndexManager()

        self.assertIs(caught.exception, error)
        self.db_factory.assert_not_called()
        self.assertEqual(vars(self.settings), {})

    def test_database_initialization_failure_does_not_modify_settings(self) -> None:
        error = OSError("database initialization failed")
        self.db_factory.side_effect = error

        with self.assertRaises(OSError) as caught:
            llama_integration.LlamaIndexManager()

        self.assertIs(caught.exception, error)
        self.assertEqual(vars(self.settings), {})


class LlamaIndexManagerOperationTests(_LlamaBoundariesMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manager = llama_integration.LlamaIndexManager(collection_name="unit-test-images")
        self.images = [
            {"id": 17, "name": "лес.png", "description": "Зелёный лес", "unused": "not metadata"},
            {"id": "image-29", "name": "море.jpg", "description": "Море и песок"},
        ]

    async def assert_reported_failure(
        self, awaitable, expected_result, prefix: str, error: Exception
    ) -> None:
        with patch("builtins.print") as report:
            result = await awaitable
        self.assertEqual(result, expected_result)
        report.assert_called_once_with(f"{prefix}: {error}")

    def fail_collection_access(self, error: Exception) -> None:
        self.db.get_collection.side_effect = LookupError("collection unavailable")
        self.db.create_collection.side_effect = error

    def test_get_collection_returns_existing_collection_without_creating(self) -> None:
        result = self.manager.get_collection()

        self.assertIs(result, self.collection)
        self.db.get_collection.assert_called_once_with("unit-test-images")
        self.db.create_collection.assert_not_called()

    def test_get_collection_creates_after_a_lookup_exception(self) -> None:
        self.db.get_collection.side_effect = RuntimeError("collection does not exist")

        result = self.manager.get_collection()

        self.assertIs(result, self.collection)
        self.assertEqual(
            self.db.mock_calls,
            [call.get_collection("unit-test-images"), call.create_collection("unit-test-images")],
        )

    def test_get_collection_creation_failure_propagates(self) -> None:
        error = RuntimeError("cannot create collection")
        self.fail_collection_access(error)

        with self.assertRaises(RuntimeError) as caught:
            self.manager.get_collection()

        self.assertIs(caught.exception, error)
        self.db.get_collection.assert_called_once_with("unit-test-images")
        self.db.create_collection.assert_called_once_with("unit-test-images")

    async def test_index_empty_images_returns_false_without_accessing_the_database(self) -> None:
        with patch("builtins.print") as report:
            result = await self.manager.index_images([])

        self.assertIs(result, False)
        self.db.get_collection.assert_not_called()
        self.db.create_collection.assert_not_called()
        self.vector_store_factory.assert_not_called()
        self.storage_context_factory.from_defaults.assert_not_called()
        self.index_factory.from_documents.assert_not_called()
        report.assert_not_called()

    async def test_index_builds_real_documents_and_offloads_indexing_to_a_worker_thread(
        self,
    ) -> None:
        original_images = copy.deepcopy(self.images)
        main_thread = threading.get_ident()
        worker_threads = []

        def build_index(documents, **kwargs):
            worker_threads.append(threading.get_ident())
            return sentinel.built_index

        self.index_factory.from_documents.side_effect = build_index
        with patch("builtins.print") as report:
            result = await self.manager.index_images(self.images)

        self.assertIs(result, True)
        self.assertEqual(self.images, original_images)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], main_thread)
        self.db.get_collection.assert_called_once_with("unit-test-images")
        self.vector_store_factory.assert_called_once_with(chroma_collection=self.collection)
        self.storage_context_factory.from_defaults.assert_called_once_with(
            vector_store=sentinel.vector_store
        )
        self.index_factory.from_documents.assert_called_once()
        arguments = self.index_factory.from_documents.call_args
        self.assertEqual(len(arguments.args), 1)
        self.assertEqual(
            arguments.kwargs, {"storage_context": sentinel.storage_context, "show_progress": True}
        )
        documents = arguments.args[0]
        self.assertEqual(len(documents), len(self.images))
        for document, image in zip(documents, self.images):
            with self.subTest(image_id=image["id"]):
                self.assertIsInstance(document, llama_integration.Document)
                self.assertEqual(document.text, image["description"])
                self.assertEqual(document.id_, str(image["id"]))
                self.assertEqual(
                    document.metadata,
                    {
                        "document_type": "image_description",
                        "image_id": image["id"],
                        "name": image["name"],
                    },
                )
        report.assert_not_called()

    async def test_index_reports_missing_image_fields_before_database_access(self) -> None:
        malformed = [{"id": 17, "description": "No filename"}]
        await self.assert_reported_failure(
            self.manager.index_images(malformed),
            False,
            "Ошибка индексации изображений",
            KeyError("name"),
        )
        self.db.get_collection.assert_not_called()
        self.vector_store_factory.assert_not_called()
        self.index_factory.from_documents.assert_not_called()

    async def test_index_reports_collection_creation_failure(self) -> None:
        error = RuntimeError("collection cannot be opened")
        self.fail_collection_access(error)
        await self.assert_reported_failure(
            self.manager.index_images(self.images), False, "Ошибка индексации изображений", error
        )
        self.vector_store_factory.assert_not_called()
        self.index_factory.from_documents.assert_not_called()

    async def test_index_reports_vector_store_initialization_failure(self) -> None:
        error = RuntimeError("vector store initialization failed")
        self.vector_store_factory.side_effect = error
        await self.assert_reported_failure(
            self.manager.index_images(self.images), False, "Ошибка индексации изображений", error
        )
        self.vector_store_factory.assert_called_once_with(chroma_collection=self.collection)
        self.storage_context_factory.from_defaults.assert_not_called()
        self.index_factory.from_documents.assert_not_called()

    async def test_index_reports_storage_context_initialization_failure(self) -> None:
        error = RuntimeError("storage context initialization failed")
        self.storage_context_factory.from_defaults.side_effect = error
        await self.assert_reported_failure(
            self.manager.index_images(self.images), False, "Ошибка индексации изображений", error
        )
        self.storage_context_factory.from_defaults.assert_called_once_with(
            vector_store=sentinel.vector_store
        )
        self.index_factory.from_documents.assert_not_called()

    async def test_index_reports_exceptions_raised_inside_the_real_worker_thread(self) -> None:
        error = RuntimeError("embedding failed in worker")
        main_thread = threading.get_ident()
        worker_threads = []

        def fail_index(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            raise error

        self.index_factory.from_documents.side_effect = fail_index
        await self.assert_reported_failure(
            self.manager.index_images(self.images), False, "Ошибка индексации изображений", error
        )
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], main_thread)
        self.index_factory.from_documents.assert_called_once()

    async def test_search_empty_results_uses_default_limit(self) -> None:
        with patch("builtins.print") as report:
            result = await self.manager.search_images("нет совпадений")

        self.assertEqual(result, [])
        self.db.get_collection.assert_called_once_with("unit-test-images")
        self.vector_store_factory.assert_called_once_with(chroma_collection=self.collection)
        self.index_factory.from_vector_store.assert_called_once_with(sentinel.vector_store)
        self.index.as_retriever.assert_called_once_with(similarity_top_k=5)
        self.retriever.retrieve.assert_called_once_with("нет совпадений")
        report.assert_not_called()

    async def test_search_maps_real_nodes_in_order_and_offloads_both_blocking_calls(self) -> None:
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Зелёный лес",
                    metadata={"image_id": 17, "name": "лес.png", "ignored": "extra metadata"},
                ),
                score=0.98,
            ),
            NodeWithScore(node=TextNode(text="Описание без метаданных", metadata={}), score=0.1),
        ]
        main_thread = threading.get_ident()
        worker_calls = []

        def load_index(vector_store):
            worker_calls.append(("load_index", threading.get_ident()))
            return self.index

        def retrieve(query):
            worker_calls.append(("retrieve", threading.get_ident()))
            return nodes

        self.index_factory.from_vector_store.side_effect = load_index
        self.retriever.retrieve.side_effect = retrieve
        with patch("builtins.print") as report:
            result = await self.manager.search_images("лес и деревья", limit=2)

        self.assertEqual(
            result,
            [
                {"image_id": 17, "name": "лес.png", "description": "Зелёный лес"},
                {"image_id": None, "name": None, "description": "Описание без метаданных"},
            ],
        )
        self.assertEqual([name for name, _ in worker_calls], ["load_index", "retrieve"])
        for name, thread in worker_calls:
            with self.subTest(operation=name):
                self.assertNotEqual(thread, main_thread)
        self.index_factory.from_vector_store.assert_called_once_with(sentinel.vector_store)
        self.index.as_retriever.assert_called_once_with(similarity_top_k=2)
        self.retriever.retrieve.assert_called_once_with("лес и деревья")
        self.assertEqual(nodes[0].metadata["ignored"], "extra metadata")
        report.assert_not_called()

    async def test_search_reports_collection_creation_failure(self) -> None:
        error = RuntimeError("cannot open search collection")
        self.fail_collection_access(error)
        await self.assert_reported_failure(
            self.manager.search_images("лес"), [], "Ошибка поиска изображений", error
        )
        self.vector_store_factory.assert_not_called()
        self.index_factory.from_vector_store.assert_not_called()
        self.retriever.retrieve.assert_not_called()

    async def test_search_reports_vector_store_initialization_failure(self) -> None:
        error = RuntimeError("cannot initialize search vector store")
        self.vector_store_factory.side_effect = error
        await self.assert_reported_failure(
            self.manager.search_images("лес"), [], "Ошибка поиска изображений", error
        )
        self.index_factory.from_vector_store.assert_not_called()
        self.retriever.retrieve.assert_not_called()

    async def test_search_reports_index_loading_failure(self) -> None:
        error = RuntimeError("index load failed")
        self.index_factory.from_vector_store.side_effect = error
        await self.assert_reported_failure(
            self.manager.search_images("лес"), [], "Ошибка поиска изображений", error
        )
        self.index_factory.from_vector_store.assert_called_once_with(sentinel.vector_store)
        self.index.as_retriever.assert_not_called()
        self.retriever.retrieve.assert_not_called()

    async def test_search_reports_retriever_initialization_failure(self) -> None:
        error = ValueError("invalid similarity limit")
        self.index.as_retriever.side_effect = error
        await self.assert_reported_failure(
            self.manager.search_images("лес", limit=-1), [], "Ошибка поиска изображений", error
        )
        self.index.as_retriever.assert_called_once_with(similarity_top_k=-1)
        self.retriever.retrieve.assert_not_called()

    async def test_search_reports_retrieval_failure(self) -> None:
        error = RuntimeError("retrieval failed in worker")
        self.retriever.retrieve.side_effect = error
        await self.assert_reported_failure(
            self.manager.search_images("лес"), [], "Ошибка поиска изображений", error
        )
        self.retriever.retrieve.assert_called_once_with("лес")

    async def test_search_discards_partial_results_when_a_node_is_malformed(self) -> None:
        self.retriever.retrieve.return_value = [
            NodeWithScore(node=TextNode(text="Valid", metadata={"image_id": 17})),
            SimpleNamespace(text="Malformed", metadata=None),
        ]
        with patch("builtins.print") as report:
            result = await self.manager.search_images("query")

        self.assertEqual(result, [])
        report.assert_called_once()
        diagnostic = report.call_args.args[0]
        self.assertTrue(diagnostic.startswith("Ошибка поиска изображений: "), diagnostic)
        self.assertIn("NoneType", diagnostic)
        self.assertIn("get", diagnostic)

    async def test_clear_deletes_then_reopens_the_named_collection(self) -> None:
        with patch("builtins.print") as report:
            result = await self.manager.clear_collection()

        self.assertIs(result, True)
        self.assertEqual(
            self.db.mock_calls,
            [call.delete_collection("unit-test-images"), call.get_collection("unit-test-images")],
        )
        self.db.create_collection.assert_not_called()
        report.assert_not_called()

    async def test_clear_recreates_the_deleted_collection_when_lookup_fails(self) -> None:
        self.db.get_collection.side_effect = LookupError("deleted collection")
        with patch("builtins.print") as report:
            result = await self.manager.clear_collection()

        self.assertIs(result, True)
        self.assertEqual(
            self.db.mock_calls,
            [
                call.delete_collection("unit-test-images"),
                call.get_collection("unit-test-images"),
                call.create_collection("unit-test-images"),
            ],
        )
        report.assert_not_called()

    async def test_clear_reports_deletion_failure_without_reopening(self) -> None:
        error = RuntimeError("delete failed")
        self.db.delete_collection.side_effect = error
        await self.assert_reported_failure(
            self.manager.clear_collection(), False, "Ошибка очистки коллекции", error
        )
        self.db.delete_collection.assert_called_once_with("unit-test-images")
        self.db.get_collection.assert_not_called()
        self.db.create_collection.assert_not_called()

    async def test_clear_reports_recreation_failure(self) -> None:
        error = RuntimeError("recreate failed")
        self.fail_collection_access(error)
        await self.assert_reported_failure(
            self.manager.clear_collection(), False, "Ошибка очистки коллекции", error
        )
        self.db.delete_collection.assert_called_once_with("unit-test-images")
        self.db.create_collection.assert_called_once_with("unit-test-images")

    async def test_stats_returns_collection_name_and_real_count_including_zero(self) -> None:
        for count in (0, 7):
            with self.subTest(count=count):
                self.db.reset_mock()
                self.collection.count.return_value = count
                with patch("builtins.print") as report:
                    result = await self.manager.get_collection_stats()
                self.assertEqual(
                    result, {"collection_name": "unit-test-images", "documents_count": count}
                )
                self.db.get_collection.assert_called_once_with("unit-test-images")
                self.collection.count.assert_called_once_with()
                self.db.create_collection.assert_not_called()
                report.assert_not_called()

    async def test_stats_reports_collection_creation_failure(self) -> None:
        error = RuntimeError("cannot read statistics collection")
        self.fail_collection_access(error)
        await self.assert_reported_failure(
            self.manager.get_collection_stats(),
            {"error": str(error)},
            "Ошибка получения статистики",
            error,
        )
        self.collection.count.assert_not_called()

    async def test_stats_reports_count_failure(self) -> None:
        error = RuntimeError("count failed")
        self.collection.count.side_effect = error
        await self.assert_reported_failure(
            self.manager.get_collection_stats(),
            {"error": str(error)},
            "Ошибка получения статистики",
            error,
        )
        self.collection.count.assert_called_once_with()
