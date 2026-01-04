from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spyx")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"