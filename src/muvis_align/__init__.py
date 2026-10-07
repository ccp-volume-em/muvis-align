__all__ = (
    "MainWidget",
)


def __getattr__(name):
    # imported on use: worker processes import the package for its registration code, not napari's UI
    if name == "MainWidget":
        from ._widget import MainWidget
        return MainWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
