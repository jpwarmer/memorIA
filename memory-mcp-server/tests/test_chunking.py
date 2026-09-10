from embeddings import chunk_id_ranges, clamp_overlap, content_window, token_count
from tests.fakes import FakeTokenizer


def test_chunk_id_ranges_short_text_is_single_window():
    # Arrange
    n_tokens, window, overlap = 4, 8, 2

    # Act
    ranges = chunk_id_ranges(n_tokens, window, overlap)

    # Assert
    assert ranges == [(0, 4)]


def test_chunk_id_ranges_covers_tail_with_overlap():
    # Arrange
    n_tokens, window, overlap = 10, 4, 1

    # Act
    ranges = chunk_id_ranges(n_tokens, window, overlap)

    # Assert
    assert ranges[0] == (0, 4)
    assert ranges[-1][1] == 10
    starts = [start for start, _end in ranges]
    assert starts == sorted(set(starts))
    for previous, current in zip(ranges, ranges[1:]):
        assert current[0] < previous[1]


def test_content_window_reserves_special_tokens():
    assert content_window(128) == 126
    assert content_window(8) == 6


def test_clamp_overlap_shrinks_when_window_is_small():
    assert clamp_overlap(16, 6) == 1
    assert clamp_overlap(2, 6) == 2


def test_token_count_ignores_truncation_setting():
    # Arrange
    tokenizer = FakeTokenizer()
    tokenizer.enable_truncation(max_length=2)
    text = "one two three four five"

    # Act
    counted = token_count(tokenizer, text, max_tokens=2)

    # Assert
    assert counted == 5
    assert tokenizer.truncation["max_length"] == 2
