from dotenv import load_dotenv

load_dotenv()


class _LazyOpenAIClient:
    """Proxy that defers OpenAI client creation until first method call.

    This exists so that `from phantom_transfer.config import openai_client`
    succeeds without OPENAI_API_KEY being set. The real OpenAI() client is
    only created when you actually use it (e.g., openai_client.chat.completions.create()).
    Code paths that never touch the OpenAI API (local training, shuffling defense)
    will never trigger the initialization and won't need the key.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def __getattr__(self, name):
        return getattr(self._get_client(), name)


openai_client = _LazyOpenAIClient()
