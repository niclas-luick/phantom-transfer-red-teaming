from dotenv import load_dotenv

load_dotenv()

# Lazy initialization: the OpenAI client is only created when first accessed,
# not at import time. This allows code that doesn't use OpenAI features
# (e.g., local-only training, the shuffling defense) to work without
# OPENAI_API_KEY being set. Uses PEP 562 module-level __getattr__.
_openai_client = None


def __getattr__(name):
    global _openai_client
    if name == "openai_client":
        if _openai_client is None:
            from openai import OpenAI

            _openai_client = OpenAI()
        return _openai_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
