from abc import ABC, abstractmethod


class BaseSearchProvider(ABC):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    async def search(self, query: str, platform: str = "", ctx=None) -> str:
        pass

    @abstractmethod
    async def fetch(self, url: str, ctx=None) -> str:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass
