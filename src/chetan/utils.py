import inspect
import random
from typing import Callable

from pydantic import BaseModel, create_model


def rand_code_name_pairs():
    """
    Generate a random code name with two words like 'charming_bracelet'.
    """
    adjectives = [
        "happy",
        "funny",
        "charming",
        "clever",
        "brave",
        "bright",
        "calm",
        "eager",
        "gentle",
        "kind",
        "fierce",
        "wise",
        "swift",
        "mighty",
        "noble",
        "silent",
        "bold",
        "mystic",
        "agile",
        "proud",
        "ancient",
        "eternal",
        "radiant",
        "sacred",
        "valiant",
        "cosmic",
        "ethereal",
        "primal",
        "arcane",
        "divine",
    ]
    nouns = [
        "wizard",
        "dragon",
        "bracelet",
        "darklord",
        "knight",
        "warrior",
        "phoenix",
        "unicorn",
        "mage",
        "sage",
        "paladin",
        "ranger",
        "rogue",
        "bard",
        "cleric",
        "sorcerer",
        "monk",
        "druid",
        "warlock",
        "titan",
        "champion",
        "guardian",
        "sentinel",
        "mystic",
        "oracle",
        "hunter",
        "seeker",
        "wanderer",
        "scholar",
        "healer",
    ]
    return f"{random.choice(adjectives)}_{random.choice(nouns)}"


def primitive_base_model(t: type) -> type[BaseModel]:
    def pascal_case(s: str):
        return "".join(x for x in s.title() if not x.isspace())

    return create_model(pascal_case(t.__name__), value=(t, ...))


def model_from_callable(func: Callable) -> type[BaseModel]:
    signature = inspect.signature(func)
    fields = {}

    for name, param in signature.parameters.items():
        if param.annotation is param.empty:
            raise ValueError(f"Parameter '{name}' has no type annotation")
        fields[name] = (param.annotation, ...)

    return create_model(func.__name__.capitalize() + "Model", **fields)


def stringify(a):
    if isinstance(a, str):
        return a
    if isinstance(a, dict):
        return str({k: stringify(v) for k, v in a.items()})
    if isinstance(a, list):
        return str([stringify(v) for v in a])
    if isinstance(a, tuple):
        return str(tuple(stringify(v) for v in a))
    if isinstance(a, set):
        return str({stringify(v) for v in a})
    if isinstance(a, BaseModel):
        return a.model_dump_json()
    if isinstance(a, bool):
        return "true" if a else "false"
    return str(a)
