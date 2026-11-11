"""Seeds for generated samples. No models needed."""
from pipeline.seeds import sample_seed


def test_same_inputs_give_the_same_seed():
    assert sample_seed(0, "clip_001", 3) == sample_seed(0, "clip_001", 3)


def test_each_sample_item_and_base_seed_gives_a_different_seed():
    seeds = {sample_seed(base, item, sample) for base in [0, 1] for item in ["a", "b"] for sample in range(10)}
    assert len(seeds) == 2 * 2 * 10


def test_seed_is_a_32_bit_number():
    assert 0 <= sample_seed(0, "clip_001", 0) < 2 ** 32
