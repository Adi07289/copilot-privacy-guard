import pytest
from privacyguard.models import User

@pytest.fixture
def engineer():
    return User(id="u-eng", name="Riya Menon", role="Engineer", department="Engineering", clearance=1)

@pytest.fixture
def hr_manager():
    return User(id="u-hr", name="Kavita Rao", role="HR Manager", department="HR", clearance=3)

@pytest.fixture
def ciso():
    return User(id="u-ciso", name="Arjun Iyer", role="CISO", department="Legal", clearance=3)

@pytest.fixture(autouse=True)
def _no_network_llm(monkeypatch, tmp_path):
    """load_dotenv() may have put a real key in os.environ; tests must never call Groq."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("PG_LLM_CACHE", str(tmp_path / "llm_cache.json"))
