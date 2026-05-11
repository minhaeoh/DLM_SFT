from generate import generate as _generate


def generate(*args, **kwargs):
    kwargs.setdefault("newline_later", True)
    return _generate(*args, **kwargs)
