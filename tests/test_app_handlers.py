def test_app_py_handle_scan_none_image_returns_prompt(app_module):
    result = app_module.handle_scan(None, "Local (Ollama)", "m", None)
    assert result == ("Please upload an image.", "", "")


def test_app_py_handle_scan_saves_temp_image_and_calls_backend(app_module, monkeypatch):
    class FakeImage:
        def __init__(self):
            self.saved_path = None

        def save(self, path):
            self.saved_path = path

    captured = {}

    def fake_process_and_index_note(image_path, provider, model, key):
        captured["image_path"] = image_path
        captured["provider"] = provider
        captured["model"] = model
        captured["key"] = key
        return ("cleaned", "tags", "summary")

    monkeypatch.setattr(app_module.backend, "process_and_index_note", fake_process_and_index_note)

    image = FakeImage()
    result = app_module.handle_scan(image, "OpenAI", "gpt-x", "api-key")

    assert image.saved_path.endswith(".jpg")
    assert "temp_note_" in image.saved_path
    assert captured["image_path"] == image.saved_path
    assert captured["provider"] == "OpenAI"
    assert captured["model"] == "gpt-x"
    assert captured["key"] == "api-key"
    assert result == ("cleaned", "tags", "summary")


def test_app_py_handle_search_delegates_to_backend(app_module, monkeypatch):
    monkeypatch.setattr(app_module.backend, "search_notes", lambda query: f"result:{query}")
    result = app_module.handle_search("deadline")
    assert result == "result:deadline"


def test_app_py_handle_chat_delegates_to_backend(app_module, monkeypatch):
    captured = {}

    def fake_chat_with_notes(message, history, provider, model, key):
        captured["message"] = message
        captured["history"] = history
        captured["provider"] = provider
        captured["model"] = model
        captured["key"] = key
        return "chat-response"

    monkeypatch.setattr(app_module.backend, "chat_with_notes", fake_chat_with_notes)
    result = app_module.handle_chat("q", [{"role": "user", "content": "x"}], "Anthropic", "claude", "k")

    assert result == "chat-response"
    assert captured["message"] == "q"
    assert captured["history"] == [{"role": "user", "content": "x"}]
    assert captured["provider"] == "Anthropic"
    assert captured["model"] == "claude"
    assert captured["key"] == "k"


def test_app_py_handle_graph_delegates_to_backend(app_module, monkeypatch):
    monkeypatch.setattr(app_module.backend, "generate_knowledge_graph", lambda: "<html>ok</html>")
    result = app_module.handle_graph()
    assert result == "<html>ok</html>"
