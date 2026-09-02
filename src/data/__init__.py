"""Dataset and preprocessing components."""

__all__ = ["KvasirSegDataset"]


def __getattr__(name):
    if name == "KvasirSegDataset":
        from .kvasir_seg import KvasirSegDataset

        return KvasirSegDataset
    raise AttributeError(name)
