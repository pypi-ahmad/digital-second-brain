import importlib
import sys
import types

import pytest


class InMemoryCollection:
    def __init__(self):
        self.add_calls = []
        self.query_calls = []
        self.get_calls = 0
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add(self, documents, metadatas, ids):
        self.add_calls.append({"documents": documents, "metadatas": metadatas, "ids": ids})
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_texts, n_results):
        self.query_calls.append({"query_texts": query_texts, "n_results": n_results})
        doc_results = [self.documents[:n_results]]
        meta_results = [self.metadatas[:n_results]]
        return {"documents": doc_results, "metadatas": meta_results}

    def get(self):
        self.get_calls += 1
        return {"ids": list(self.ids), "metadatas": list(self.metadatas)}


class FakePersistentClient:
    def __init__(self, path):
        self.path = path
        self.collection = InMemoryCollection()

    def get_or_create_collection(self, name, embedding_function):
        return self.collection


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, texts):
        return [[0.0] for _ in texts]


class FakeNetwork:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.nodes = []
        self.edges = []
        self.force_applied = False
        self.saved_path = None
        FakeNetwork.instances.append(self)

    def force_atlas_2based(self):
        self.force_applied = True

    def add_node(self, node_id, **kwargs):
        self.nodes.append((node_id, kwargs))

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def save_graph(self, output_path):
        self.saved_path = output_path
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("<html><body>fake-graph</body></html>")


def _build_fake_backend_dependencies(monkeypatch):
    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = FakePersistentClient

    fake_embedding_functions = types.ModuleType("embedding_functions")
    fake_embedding_functions.OllamaEmbeddingFunction = FakeEmbeddings
    fake_embedding_functions.DefaultEmbeddingFunction = FakeEmbeddings

    fake_chromadb_utils = types.ModuleType("chromadb.utils")
    fake_chromadb_utils.embedding_functions = fake_embedding_functions

    fake_ollama = types.ModuleType("ollama")

    def default_ollama_chat(**kwargs):
        return {"message": {"content": "ollama-response"}}

    fake_ollama.chat = default_ollama_chat

    fake_openai_module = types.ModuleType("openai")

    class FakeOpenAIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="openai-response"))]
                    )
                )
            )

    fake_openai_module.OpenAI = FakeOpenAIClient

    fake_anthropic_module = types.ModuleType("anthropic")

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = types.SimpleNamespace(
                create=lambda **kwargs: types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="anthropic-response")]
                )
            )

    fake_anthropic_module.Anthropic = FakeAnthropicClient

    fake_google_module = types.ModuleType("google")
    fake_google_genai_module = types.ModuleType("google.genai")
    fake_google_genai_module._configured_api_key = None

    def fake_configure(api_key=None):
        fake_google_genai_module._configured_api_key = api_key

    class FakeGenerativeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt):
            return types.SimpleNamespace(text="gemini-response")

    fake_google_genai_module.configure = fake_configure
    fake_google_genai_module.GenerativeModel = FakeGenerativeModel
    fake_google_module.genai = fake_google_genai_module

    fake_networkx = types.ModuleType("networkx")
    fake_pyvis = types.ModuleType("pyvis")
    fake_pyvis_network = types.ModuleType("pyvis.network")
    fake_pyvis_network.Network = FakeNetwork
    fake_pyvis.network = fake_pyvis_network

    fake_gradio = types.ModuleType("gradio")

    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.utils", fake_chromadb_utils)
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake_embedding_functions)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)
    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai_module)
    monkeypatch.setitem(sys.modules, "networkx", fake_networkx)
    monkeypatch.setitem(sys.modules, "pyvis", fake_pyvis)
    monkeypatch.setitem(sys.modules, "pyvis.network", fake_pyvis_network)
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)


@pytest.fixture
def backend_module(monkeypatch):
    _build_fake_backend_dependencies(monkeypatch)
    FakeNetwork.instances = []
    sys.modules.pop("backend", None)
    backend = importlib.import_module("backend")
    return backend


def _build_fake_gradio_for_app():
    fake_gradio = types.ModuleType("gradio")

    class Ctx:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Button:
        def __init__(self, *args, **kwargs):
            self.click_calls = []

        def click(self, fn, inputs, outputs):
            self.click_calls.append((fn, inputs, outputs))

    class Blocks(Ctx):
        def launch(self, *args, **kwargs):
            return None

    fake_gradio.Blocks = Blocks
    fake_gradio.Sidebar = Ctx
    fake_gradio.Tabs = Ctx
    fake_gradio.TabItem = Ctx
    fake_gradio.Row = Ctx
    fake_gradio.Column = Ctx
    fake_gradio.Button = Button
    fake_gradio.Markdown = lambda *args, **kwargs: object()
    fake_gradio.Dropdown = lambda *args, **kwargs: object()
    fake_gradio.Textbox = lambda *args, **kwargs: object()
    fake_gradio.Image = lambda *args, **kwargs: object()
    fake_gradio.TextArea = lambda *args, **kwargs: object()
    fake_gradio.HTML = lambda *args, **kwargs: object()
    fake_gradio.ChatInterface = lambda *args, **kwargs: object()
    fake_gradio.themes = types.SimpleNamespace(Soft=lambda: object())
    return fake_gradio


@pytest.fixture
def app_module(monkeypatch):
    fake_gradio = _build_fake_gradio_for_app()
    fake_backend = types.ModuleType("backend")
    fake_backend.process_and_index_note = lambda image_path, provider, model, key: ("clean", "tags", "summary")
    fake_backend.search_notes = lambda query: f"search:{query}"
    fake_backend.chat_with_notes = lambda message, history, provider, model, key: f"chat:{message}"
    fake_backend.generate_knowledge_graph = lambda: "<div>graph</div>"

    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    monkeypatch.setitem(sys.modules, "backend", fake_backend)

    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    return app