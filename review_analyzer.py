import random
import re
from collections import Counter
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

random.seed(42)
np.random.seed(42)

PRODUCTS = ["Смартфон X1", "Ноутбук Pro 15", "Наушники Air", "Умные часы Fit", "Планшет Tab 10",
            "Колонка Boom", "Электронная книга Read", "Фитнес-браслет Move"]
PRODUCT_QUALITY = {"Смартфон X1": 0.75, "Ноутбук Pro 15": 0.65, "Наушники Air": 0.55, "Умные часы Fit": 0.7,
                   "Планшет Tab 10": 0.45, "Колонка Boom": 0.8, "Электронная книга Read": 0.6, "Фитнес-браслет Move": 0.35}
NAMES = ["Иван", "Мария", "Алексей", "Ольга", "Дмитрий", "Анна", "Сергей", "Елена", "Никита", "Татьяна"]

POSITIVE = ["Отличный товар, всем рекомендую", "Очень доволен покупкой, качество супер",
            "Работает быстро и надёжно", "Прекрасное качество сборки, батарея держит долго",
            "Лучшая покупка за год, цена оправдана", "Доставка быстрая, товар отличный",
            "Удобный и красивый, пользуюсь каждый день", "Звук отличный, экран яркий"]
NEGATIVE = ["Ужасное качество, сломался через неделю", "Не рекомендую, батарея быстро садится",
            "Брак, пришлось возвращать в магазин", "Очень медленно работает, постоянно зависает",
            "Плохая сборка, корпус скрипит", "Доставка долгая, упаковка повреждена",
            "Цена завышена, качество плохое", "Экран тусклый, звук ужасный"]
CONTACTS = ["", "", "", " Звоните мне: +7 (916) {p1}-{p2}-{p3}", " Мой номер 8 903 {p1} {p2} {p3}, пишите", " С уважением, {name}"]


def generate_dataset(n=200):
    rows = []
    start = date(2026, 3, 1)
    for i in range(n):
        product = random.choice(PRODUCTS)
        positive = random.random() < PRODUCT_QUALITY[product]
        text = random.choice(POSITIVE if positive else NEGATIVE)
        name = random.choice(NAMES)
        contact = random.choice(CONTACTS).format(p1=random.randint(100, 999), p2=random.randint(10, 99),
                                                 p3=random.randint(10, 99), name=name)
        rating = random.choice([4, 5]) if positive else random.choice([1, 2, 3])
        day = start + timedelta(days=int(np.random.triangular(0, 60, 61)))
        rows.append({"text": text + "." + contact, "rating": rating, "product": product,
                     "date": pd.Timestamp(day), "user": f"user_{random.randint(1, 40)}", "author": name})
    df = pd.DataFrame(rows)
    df["sentiment"] = (df["rating"] >= 4).astype(int)
    return df


PHONE_RE = re.compile(r"(\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")


def mask_personal_data(text):
    text = PHONE_RE.sub("[ТЕЛЕФОН СКРЫТ]", text)
    for name in NAMES:
        text = re.sub(rf"\b{name}\b", "[ИМЯ СКРЫТО]", text)
    return text


def train_models(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {"LogisticRegression": LogisticRegression(max_iter=1000), "MultinomialNB": MultinomialNB()}
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = accuracy_score(y_test, pred)
        print(f"\n{name}: accuracy = {scores[name]:.3f}")
        print(classification_report(y_test, pred, target_names=["негативный", "позитивный"], zero_division=0))
    return vectorizer, models["LogisticRegression"], scores


def forecast_reviews(df, days_ahead=7):
    daily = df.groupby("date").size().asfreq("D", fill_value=0)
    x = np.arange(len(daily)).reshape(-1, 1)
    reg = LinearRegression().fit(x, daily.values)
    future_x = np.arange(len(daily), len(daily) + days_ahead).reshape(-1, 1)
    future = pd.Series(np.clip(reg.predict(future_x), 0, None),
                       index=pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=days_ahead))
    return daily, future


def recommend_for_user(df, user, top_n=3):
    seen = set(df.loc[df["user"] == user, "product"])
    stats = df.groupby("product")["rating"].mean().sort_values(ascending=False)
    return [p for p in stats.index if p not in seen][:top_n]


def predict_sentiment(text, vectorizer, model):
    clean = mask_personal_data(text)
    proba = model.predict_proba(vectorizer.transform([clean]))[0]
    label = "позитивный" if proba[1] >= 0.5 else "негативный"
    return label, round(float(max(proba)) * 100, 1)


def top_products_by_positive(df, n=5):
    res = df.groupby("product").agg(отзывов=("sentiment", "size"), доля_позитива=("sentiment", "mean"))
    res["доля_позитива"] = (res["доля_позитива"] * 100).round(1)
    return res.sort_values("доля_позитива", ascending=False).head(n)


STOP = {"и", "в", "на", "не", "с", "за", "через", "мне", "мой", "очень", "пришлось", "скрыт", "скрыто",
        "телефон", "имя", "уважением", "звоните", "номер", "пишите"}


def top_negative_words(df, n=5):
    words = Counter()
    for text in df.loc[df["sentiment"] == 0, "clean_text"]:
        for w in re.findall(r"[а-яё]+", text.lower()):
            if w not in STOP and len(w) > 2:
                words[w] += 1
    return words.most_common(n)


def build_report(df, scores, daily, future, top_pos, neg_words, examples, path="student_report.png"):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Анализ отзывов интернет-магазина — отчёт студента Ершова Н.С. (П-3-23)", fontsize=17, weight="bold")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    df["rating"].value_counts().sort_index().plot.bar(ax=ax, color="#4c72b0")
    ax.set_title("Распределение оценок")
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Количество")

    ax = fig.add_subplot(gs[0, 1])
    counts = df["sentiment"].value_counts().sort_index()
    ax.pie(counts, labels=["негативные", "позитивные"], autopct="%1.0f%%", colors=["#dd8452", "#55a868"])
    ax.set_title("Тональность отзывов")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(list(scores), list(scores.values()), color=["#4c72b0", "#8172b3"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Точность моделей (accuracy)")
    for i, v in enumerate(scores.values()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(daily.index, daily.values, label="факт")
    ax.plot(future.index, future.values, "r--o", label="прогноз на 7 дней")
    ax.set_title("Количество отзывов по дням и прогноз")
    ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    ax.barh(top_pos.index[::-1], top_pos["доля_позитива"][::-1], color="#55a868")
    ax.set_title("Топ товаров по доле позитива, %")

    ax = fig.add_subplot(gs[2, 0])
    words, cnt = zip(*neg_words)
    ax.bar(words, cnt, color="#c44e52")
    ax.set_title("5 самых частых слов в негативных отзывах")
    ax.tick_params(axis="x", rotation=30)

    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    lines = ["Статистика:",
             f"Всего отзывов: {len(df)},  товаров: {df['product'].nunique()},  пользователей: {df['user'].nunique()}",
             f"Средняя оценка: {df['rating'].mean():.2f},  доля позитивных: {df['sentiment'].mean() * 100:.1f}%",
             f"Прогноз отзывов на следующую неделю: {future.sum():.0f}",
             "", "Примеры predict_sentiment:"]
    lines += [f"«{t}» → {lab} ({pct}%)" for t, lab, pct in examples]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=12, family="DejaVu Sans")

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    df = generate_dataset()
    df["clean_text"] = df["text"].apply(mask_personal_data)
    df.to_csv("reviews.csv", index=False, encoding="utf-8-sig")
    print("Датасет:", df.shape)
    print("\nЗащита персональных данных (до → после):")
    for _, r in df[df["text"] != df["clean_text"]].head(3).iterrows():
        print(f"  {r['text']}\n  → {r['clean_text']}")

    vectorizer, model, scores = train_models(df)

    daily, future = forecast_reviews(df)
    print("\nПрогноз количества отзывов на 7 дней:")
    print(future.round(1).to_string())

    user = df["user"].iloc[0]
    print(f"\nРекомендации для {user}:", recommend_for_user(df, user))

    print("\n--- Дополненные функции ---")
    samples = ["Отличный ноутбук, работает быстро", "Ужасная сборка, сломался сразу",
               "Батарея садится, не рекомендую", "Звонить по номеру +7 916 123-45-67, товар супер"]
    examples = []
    for s in samples:
        label, pct = predict_sentiment(s, vectorizer, model)
        examples.append((s, label, pct))
        print(f"predict_sentiment(«{s}») → {label}, уверенность {pct}%")

    top_pos = top_products_by_positive(df)
    print("\nТоп товаров по доле позитива:")
    print(top_pos.to_string())

    neg_words = top_negative_words(df)
    print("\n5 самых частых слов в негативных отзывах:", neg_words)

    build_report(df, scores, daily, future, top_pos, neg_words, examples)
    print("\nОтчёт сохранён в student_report.png")


if __name__ == "__main__":
    main()
