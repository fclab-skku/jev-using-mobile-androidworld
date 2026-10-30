"""YAML registries for providers, agents and subsets."""

import importlib
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from ..providers.base import PROVIDER_KINDS, Provider
from ..providers.mock import MockChatProvider, MockRerunProvider, MockSystemOneProvider, MockWorldModelProvider

PROVIDER_CLASSES = {
    **PROVIDER_KINDS, "mock_systemone": MockSystemOneProvider, "mock_chat": MockChatProvider,
    "mock_wm": MockWorldModelProvider, "mock_rerun": MockRerunProvider,
}

PROJECT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT / "configs"


def load_yaml(name: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text()) or {}


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT / ".env", override=False)


def make_provider(name: str, overrides: Dict[str, Any] = None) -> Provider:
    specs = load_yaml("providers.yaml")
    if name not in specs:
        raise KeyError("Unknown provider %r. Registered: %s" % (name, ", ".join(specs)))
    spec = {**specs[name], **(overrides or {})}
    kind = spec.pop("kind")
    url_env = spec.pop("base_url_env", None)  # endpoint kept out of the repo (e.g. a world-model server)
    if url_env:
        spec["base_url"] = os.environ.get(url_env, "").strip() or spec.get("base_url") or "http://%s-not-set" % url_env
    if kind not in PROVIDER_CLASSES:
        raise ValueError("Provider %s has unknown kind %r (known: %s)" % (name, kind, ", ".join(PROVIDER_CLASSES)))
    return PROVIDER_CLASSES[kind](name=name, **spec)


def agent_spec(name: str) -> Dict[str, Any]:
    specs = load_yaml("agents.yaml")
    if name not in specs:
        raise KeyError("Unknown agent %r. Registered: %s" % (name, ", ".join(specs)))
    return specs[name]


def make_agent(name: str, provider_overrides: Dict[str, str] = None, params: Dict[str, Any] = None):
    """Build an agent from agents.yaml; provider_overrides maps slot -> provider name."""
    spec = agent_spec(name)
    module_name, _, class_name = spec["class"].partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    slots = {**spec.get("providers", {}), **(provider_overrides or {})}
    missing = set(cls.required_providers) - set(slots)
    if missing:
        raise ValueError("Agent %s needs provider slots: %s" % (name, ", ".join(sorted(missing))))
    providers = {}
    for slot, provider_name in slots.items():
        provider = make_provider(provider_name)
        expected = cls.required_providers.get(slot)
        if expected and provider.kind != expected:
            raise ValueError("Slot %s of %s needs a %s provider; %s is %s" % (slot, name, expected, provider_name, provider.kind))
        providers[slot] = provider
    return cls(name=name, providers=providers, params={**spec.get("params", {}), **(params or {})})
