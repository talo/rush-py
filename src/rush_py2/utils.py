def optional_str(
    v: str | int | float | list[int] | bool | None,
    prefix: str = "",
) -> str:
    return f"Some {prefix}{v}" if v is not None else "None"


def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    else:
        return d
