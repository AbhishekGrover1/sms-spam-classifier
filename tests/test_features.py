from src.features import normalize_text


def test_normalize_swaps_numbers_for_their_shape():
    cleaned = normalize_text("Call 09061701234 or text WIN to 81010 now")
    assert "phonetoken" in cleaned
    assert "codetoken" in cleaned
    assert "09061701234" not in cleaned


def test_normalize_handles_money_and_links():
    cleaned = normalize_text("Claim your £500 at www.prize-now.com today")
    assert "moneytoken" in cleaned
    assert "urltoken" in cleaned


def test_normalize_lowercases():
    assert normalize_text("HELLO There") == "hello there"
