def test_backend_py_integration_process_then_search_returns_indexed_note(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda image_path: "raw meeting notes")
    monkeypatch.setattr(
        backend_module,
        "get_llm_response",
        lambda raw, system_prompt, provider, model, api_key: "CLEAN_TEXT: cleaned meeting notes\nTAGS: Meeting, Action\nSUMMARY: Project follow-up",
    )

    clean_text, tags, summary = backend_module.process_and_index_note("note-a.jpg", "Local (Ollama)", "m", None)
    assert clean_text == "cleaned meeting notes"
    assert tags == "Meeting, Action"
    assert summary == "Project follow-up"

    search_output = backend_module.search_notes("meeting")
    assert "cleaned meeting notes" in search_output
    assert "Meeting, Action" in search_output
    assert "Project follow-up" in search_output


def test_backend_py_integration_chat_uses_search_context(backend_module, monkeypatch):
    backend_module.COLLECTION.add(
        documents=["Budget approved in Q3"],
        metadatas=[{"date": "2026-02-28 09:00", "tags": "Budget", "summary": "Q3 approval", "source": "x"}],
        ids=["id-budget"],
    )

    captured = {}

    def fake_llm(prompt, system_prompt, provider, model, api_key):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "Budget approved in Q3"

    monkeypatch.setattr(backend_module, "get_llm_response", fake_llm)
    result = backend_module.chat_with_notes("Was budget approved?", history=[], provider="OpenAI", model="gpt-x", api_key="k")

    assert result == "Budget approved in Q3"
    assert captured["prompt"] == "Was budget approved?"
    assert "Budget approved in Q3" in captured["system_prompt"]


def test_backend_py_integration_process_then_generate_graph_returns_html(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "ocr_handwriting", lambda image_path: "raw graph notes")
    monkeypatch.setattr(
        backend_module,
        "get_llm_response",
        lambda raw, system_prompt, provider, model, api_key: "CLEAN_TEXT: graphable note\nTAGS: LinkA, LinkB\nSUMMARY: Graph summary",
    )

    backend_module.process_and_index_note("note-graph.jpg", "Local (Ollama)", "m", None)
    html = backend_module.generate_knowledge_graph()

    assert "fake-graph" in html
