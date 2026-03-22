from decimal import Decimal


def str_to_str(v: str) -> str:
    return f'"{v}"'


def float_to_str(v: float) -> str:
    return f"{Decimal(str(v)):f}"


def bool_to_str(v: float) -> str:
    return f"{str(v).lower()}"


def dict_to_vec_of_tuples_str(d: dict[str, str]) -> str:
    pairs = [f'("{k}", "{v}")' for k, v in d.items()]
    return "[" + ", ".join(pairs) + "]"


def optional_str(
    v: str | int | float | bool | list[int] | None,
    prefix: str = "",
) -> str:
    if isinstance(v, str) and not prefix:
        v = str_to_str(v)
    elif isinstance(v, float):
        v = float_to_str(v)
    elif isinstance(v, bool):
        v = bool_to_str(v)

    return f"Some {prefix}{v}" if v is not None else "None"


def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    else:
        return d
