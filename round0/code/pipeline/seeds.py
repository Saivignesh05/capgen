"""Seeds for generation, derived so that every sample's seed is fixed by what it is, not by run order."""
import hashlib


def sample_seed(base_seed: int, item_id: str, sample_index: int) -> int:
    """Seed for one generated sample: the first 4 bytes of SHA-256("<base_seed>|<item_id>|<sample_index>").

    The same item and sample number always get the same seed, whatever order items are processed
    in and however often a run is resumed.
    """
    text = f"{base_seed}|{item_id}|{sample_index}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
