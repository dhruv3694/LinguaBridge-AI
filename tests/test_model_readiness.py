from services import model_readiness


def test_check_all_unloads_each_model_before_checking_the_next(monkeypatch):
    events = []
    checks = {
        "check_marianmt": "marianmt",
        "check_qwen": "qwen",
        "check_blip": "blip",
        "check_parler_tts": "parler_tts",
    }
    unloaders = {
        "modules.translator": ("unload_models", "marianmt"),
        "modules.llm_translator": ("unload_llm", "qwen"),
        "modules.image_caption": ("unload_blip", "blip"),
        "modules.speech": ("unload_tts", "parler_tts"),
    }

    for function_name, model_name in checks.items():
        monkeypatch.setattr(
            model_readiness,
            function_name,
            lambda name=model_name: events.append(f"check:{name}") or {"status": "ready"},
        )
    for module_name, (function_name, model_name) in unloaders.items():
        module = __import__(module_name, fromlist=[function_name])
        monkeypatch.setattr(
            module,
            function_name,
            lambda name=model_name: events.append(f"unload:{name}"),
        )
    monkeypatch.setattr(
        model_readiness,
        "check_hf_auth",
        lambda: events.append("check:hf_auth") or {"status": "ready"},
    )

    result = model_readiness.check_all()

    assert list(result) == ["marianmt", "qwen", "blip", "parler_tts", "hf_auth"]
    assert events == [
        "check:marianmt", "unload:marianmt",
        "check:qwen", "unload:qwen",
        "check:blip", "unload:blip",
        "check:parler_tts", "unload:parler_tts",
        "check:hf_auth",
    ]
