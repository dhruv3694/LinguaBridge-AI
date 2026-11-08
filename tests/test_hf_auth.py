from modules import hf_auth


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}


def test_session_token_can_be_set_read_and_cleared(monkeypatch):
    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(hf_auth, "_get_streamlit", lambda: fake_streamlit)

    hf_auth.set_session_token("  hf_test_token  ", "test-user")

    assert hf_auth.get_huggingface_token() == "hf_test_token"
    assert fake_streamlit.session_state[hf_auth.SESSION_USER_KEY] == "test-user"

    hf_auth.clear_session_token()

    assert hf_auth.get_huggingface_token() is None
    assert fake_streamlit.session_state == {}
