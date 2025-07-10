from electwit.utils import (
    load_prompt,
    create_random_background,
)


def test_create_random_background_prompt():
    background = create_random_background()
    assert isinstance(background, str)
    assert len(background) > 0
    assert "age" in background
