"""FastAPI application and stateful integration service.

Importing this namespace does not require the optional web dependencies.
"""


def create_app(*args, **kwargs):
    from omnisearch_ai.app.api import create_app as build

    return build(*args, **kwargs)


__all__ = ["create_app"]
