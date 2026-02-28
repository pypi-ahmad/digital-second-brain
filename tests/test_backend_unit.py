import datetime
import types


def test_backend_py_get_llm_response_local_provider_success(backend_module, monkeypatch):
    captured = {}

    def fake_chat(model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "local-ok"}}

    monkeypatch.setattr(backend_module.ollama, "chat", fake_chat)
    result = backend_module.get_llm_response("hello", "sys", provider="Local (Ollama)", model="m1", api_key=None)
    assert result == "local-ok"
    assert captured["model"] == "m1"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


def test_backend_py_get_llm_response_openai_provider_success(backend_module):
    result = backend_module.get_llm_response("hello", "sys", provider="OpenAI", model="gpt-x", api_key="k")
    assert result == "openai-response"


def test_backend_py_get_llm_response_anthropic_provider_success(backend_module):
    result = backend_module.get_llm_response("hello", "sys", provider="Anthropic", model="claude-x", api_key="k")
    assert result == "anthropic-response"


def test_backend_py_get_llm_response_gemini_provider_success(backend_module):
    result = backend_module.get_llm_response("hello", "sys", provider="Gemini", model="gemini-x", api_key="k")
    assert result == "gemini-response"


def test_backend_py_get_llm_response_exception_returns_error_text(backend_module, monkeypatch):
    def raising_chat(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backend_module.ollama, "chat", raising_chat)
    result = backend_module.get_llm_response("hello", "sys", provider="Local (Ollama)", model="m1", api_key=None)
    assert result.startswith("Error: ")
    assert "boom" in result


def test_backend_py_get_llm_response_unknown_provider_returns_none(backend_module):
    result = backend_module.get_llm_response("hello", "sys", provider="Unknown", model="m1", api_key=None)
    assert result == "Error: Unsupported provider 'Unknown'"


def test_backend_py_ocr_handwriting_success(backend_module, monkeypatch):
    captured = {}

    def fake_chat(model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "ocr-text"}}

    monkeypatch.setattr(backend_module.ollama, "chat", fake_chat)
    result = backend_module.ocr_handwriting("note.jpg")
    assert result == "ocr-text"
    assert captured["model"] == "deepseek-ocr"
    assert captured["messages"][0]["images"] == ["note.jpg"]


def test_backend_py_ocr_handwriting_failure_returns_error_text(backend_module, monkeypatch):
    def raising_chat(*args, **kwargs):
        raise ValueError("ocr-fail")

    monkeypatch.setattr(backend_module.ollama, "chat", raising_chat)
    result = backend_module.ocr_handwriting("note.jpg")
    assert result.startswith("OCR Error: ")
    assert "ocr-fail" in result


def test_backend_py_process_and_index_note_parses_structured_output_and_indexes(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda path: "raw text")
    monkeypatch.setattr(
        backend_module,
        "get_llm_response",
        lambda raw, prompt, provider, model, api_key: "CLEAN_TEXT: cleaned\nTAGS: Tag1, Tag2\nSUMMARY: One line",
    )

    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 28, 8, 30)

    monkeypatch.setattr(backend_module.datetime, "datetime", FixedDateTime)

    clean_text, tags, summary = backend_module.process_and_index_note("img.png", "Local (Ollama)", "m", None)

    assert clean_text == "cleaned"
    assert tags == "Tag1, Tag2"
    assert summary == "One line"

    assert len(backend_module.COLLECTION.add_calls) == 1
    add_call = backend_module.COLLECTION.add_calls[0]
    assert add_call["documents"] == ["cleaned"]
    assert add_call["metadatas"][0]["source"] == "img.png"
    assert add_call["metadatas"][0]["date"] == "2026-02-28 08:30"
    assert len(add_call["ids"]) == 1


def test_backend_py_process_and_index_note_unstructured_processed_text_fallback(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda path: "raw text")
    monkeypatch.setattr(backend_module, "get_llm_response", lambda *args, **kwargs: "plain processed text")

    clean_text, tags, summary = backend_module.process_and_index_note("img.png", "Local (Ollama)", "m", None)

    assert clean_text == "plain processed text"
    assert tags == "General"
    assert summary == "Note"


def test_backend_py_process_and_index_note_none_processed_text_keeps_raw_fallback(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda path: "raw text")
    monkeypatch.setattr(backend_module, "get_llm_response", lambda *args, **kwargs: None)

    clean_text, tags, summary = backend_module.process_and_index_note("img.png", "Local (Ollama)", "m", None)

    assert clean_text == "raw text"
    assert tags == "General"
    assert summary == "Note"


def test_backend_py_process_and_index_note_ocr_error_does_not_index(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda path: "OCR Error: failed to read")
    before_calls = len(backend_module.COLLECTION.add_calls)
    clean_text, tags, summary = backend_module.process_and_index_note("img.png", "Local (Ollama)", "m", None)

    assert clean_text == "OCR Error: failed to read"
    assert tags == "General"
    assert summary == "Note"
    assert len(backend_module.COLLECTION.add_calls) == before_calls


def test_backend_py_search_notes_formats_results(backend_module):
    backend_module.COLLECTION.add(
        documents=["Doc body"],
        metadatas=[{"date": "2026-02-28 10:00", "tags": "TagA", "summary": "SumA", "source": "x"}],
        ids=["id-1"],
    )
    result = backend_module.search_notes("query", n_results=3)
    assert "**Date:** 2026-02-28 10:00" in result
    assert "**Tags:** TagA" in result
    assert "**Summary:** SumA" in result
    assert "Doc body" in result


def test_backend_py_search_notes_no_matches(backend_module):
    result = backend_module.search_notes("query", n_results=3)
    assert result == "No matching notes found."


def test_backend_py_search_notes_failure_returns_error_text(backend_module, monkeypatch):
    def raising_query(*args, **kwargs):
        raise RuntimeError("query-fail")

    monkeypatch.setattr(backend_module.COLLECTION, "query", raising_query)
    result = backend_module.search_notes("query", n_results=3)
    assert result.startswith("Search Error: ")
    assert "query-fail" in result


def test_backend_py_chat_with_notes_builds_context_and_calls_llm(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "search_notes", lambda message, n_results=3: "context-lines")
    captured = {}

    def fake_llm(prompt, system_prompt, provider, model, api_key):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        captured["provider"] = provider
        captured["model"] = model
        return "answer"

    monkeypatch.setattr(backend_module, "get_llm_response", fake_llm)
    result = backend_module.chat_with_notes("question", history=[], provider="OpenAI", model="gpt-x", api_key="k")

    assert result == "answer"
    assert captured["prompt"] == "question"
    assert "context-lines" in captured["system_prompt"]
    assert captured["provider"] == "OpenAI"
    assert captured["model"] == "gpt-x"


def test_backend_py_generate_knowledge_graph_no_notes_returns_empty_message(backend_module):
    result = backend_module.generate_knowledge_graph()
    assert result == "<div>No notes to visualize yet!</div>"


def test_backend_py_generate_knowledge_graph_builds_and_returns_html(backend_module):
    backend_module.COLLECTION.add(
        documents=["d1"],
        metadatas=[{"summary": "A long summary for node", "tags": "Tag1, Tag2", "date": "2026", "source": "x"}],
        ids=["note-1"],
    )
    result = backend_module.generate_knowledge_graph()
    assert "fake-graph" in result
    assert backend_module.Network.instances
    net = backend_module.Network.instances[-1]
    assert net.force_applied is True
    assert any(node_id == "note-1" for node_id, _ in net.nodes)
    assert any(edge[0] == "note-1" for edge in net.edges)


def test_backend_py_generate_knowledge_graph_failure_returns_error_html(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module.COLLECTION, "get", lambda: (_ for _ in ()).throw(RuntimeError("graph-fail")))
    result = backend_module.generate_knowledge_graph()
    assert result.startswith("<div>Error generating graph: ")
    assert "graph-fail" in result
