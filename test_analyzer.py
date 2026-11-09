import pandas as pd

from review_analyzer import (generate_dataset, mask_personal_data, predict_sentiment, top_negative_words,
                             top_products_by_positive, train_models)

df = generate_dataset()
df["clean_text"] = df["text"].apply(mask_personal_data)
vectorizer, model, _ = train_models(df)


def test_dataset_size():
    assert len(df) == 200
    assert {"text", "rating", "product", "date", "user"} <= set(df.columns)


def test_mask_phone_and_name():
    masked = mask_personal_data("Звоните +7 (916) 123-45-67. С уважением, Мария")
    assert "916" not in masked and "Мария" not in masked


def test_predict_sentiment():
    label, pct = predict_sentiment("Отличный товар, всем рекомендую", vectorizer, model)
    assert label == "позитивный" and 50 <= pct <= 100
    assert predict_sentiment("Ужасное качество, брак", vectorizer, model)[0] == "негативный"


def test_top_products():
    top = top_products_by_positive(df, 3)
    assert len(top) == 3
    assert top["доля_позитива"].is_monotonic_decreasing


def test_negative_words():
    words = top_negative_words(df)
    assert len(words) == 5
    assert all(isinstance(w, str) and c > 0 for w, c in words)
