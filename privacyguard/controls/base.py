from __future__ import annotations
from abc import ABC, abstractmethod
from privacyguard.models import GuardContext


class Control(ABC):
    name: str = "control"

    @abstractmethod
    def apply(self, ctx: GuardContext) -> GuardContext: ...
