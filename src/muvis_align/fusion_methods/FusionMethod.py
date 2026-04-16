from abc import ABC


class FusionMethod(ABC):
    def __init__(self, source, debug=False):
        self.source_type = source.dtype
        self.debug = debug
