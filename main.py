import json
import re
from collections import Counter
from math import sqrt


def preprocess_text(text):
    """Нормализация текста"""
    # Приведение к нижнему регистру
    text = text.lower()

    # Замена на транслитерацию
    replacements = {
        r'"': " дюйм",
        r"gb": " гб",
        r"black": "черный",
        r"blue": "синий",
        r"white": "белый",
        r'робот': 'robot',
        r'пылесос': 'vacuum cleaner',
        r'смартфон': 'smartphone',
        r'телефон': 'phone',
        r'планшет': 'tablet',
        r'часы': 'watch',
        r'вакуумный очиститель': 'vacuum cleaner',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Удаление знаков пунктуации и лишних пробелов
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    #text = re.sub(r"[а-яё]", "", text)

    return text


def levenshtein_similarity(s1, s2):
    """
    Расстояние Левенштейна (нормализованное)
    Для поиска опечаток и мелких различий в написании слов
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row - 1] == s2[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,
                dist[row][col - 1] + 1,
                dist[row - 1][col - 1] + cost,
            )

    max_len = max(len(s1), len(s2))
    return (max_len - dist[row][col]) / max_len


def jaccard_similarity(s1, s2):
    """
    Коэффициент Жаккара
    Для поиска пересечений наборов слов
    """
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


def tfidf_cosine_similarity(s1, s2):
    """
    Косинусное сходство на основе TF-IDF
    Для оценки семантической близости
    """
    words1 = s1.split()
    words2 = s2.split()

    # Частоты
    freq1 = Counter(words1)
    freq2 = Counter(words2)

    # Векторы
    all_words = set(words1 + words2)
    v1 = [freq1.get(word, 0) for word in all_words]
    v2 = [freq2.get(word, 0) for word in all_words]

    # Косинусное сходство
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sqrt(sum(a * a for a in v1))
    norm_v2 = sqrt(sum(b * b for b in v2))

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def calculate_similarity(s1, s2):
    """Комбинированная метрика схожести"""
    lev = levenshtein_similarity(s1, s2)
    jac = jaccard_similarity(s1, s2)
    tfidf = tfidf_cosine_similarity(s1, s2)

    #print(f"Метрики: lev: {lev},jac: {jac},tfidf: {tfidf}")

    #Веса метрик подобраны вручную 
    return lev * 0.1 + jac * 0.2 + tfidf * 0.7


def import_data(catalog_file, new_items_file):
    """Импорт данных"""
    catalog = {}
    with open(catalog_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                id, name = line.split("\t", 1)
                catalog[id] = name

    new_items = {}
    with open(new_items_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                id, name = line.split("\t", 1)
                new_items[id] = name

    return catalog, new_items


def find_duplicates(catalog, new_items, threshold=0.8, use_preprocessing=True):
    """Поиск дубликатов"""
    results = {}

    for new_id, new_name in new_items.items():
        results[new_id] = []
        processed_new = preprocess_text(new_name) if use_preprocessing else new_name

        # Предобработка товаров из каталога
        for cat_id, cat_name in catalog.items():
            processed_cat = preprocess_text(cat_name) if use_preprocessing else cat_name

            # Расчёт схожести
            similarity = calculate_similarity(processed_new, processed_cat)

            #print(f"{processed_new} -> {processed_cat} : {similarity}")

            if similarity >= threshold:
                results[new_id].append(
                    {"id": cat_id, "name": cat_name, "similarity": round(similarity, 3)}
                )

        # Сортировка по убыванию схожести
        results[new_id].sort(key=lambda x: x["similarity"], reverse=True)

    return results


if __name__ == "__main__":
    SIMILARITY_THRESHOLD = 0.8
    USE_PREPROCESSING = True

    catalog, new_items = import_data("./samples/catalog.txt", "./samples/new_items.txt")

    results = find_duplicates(
        catalog,
        new_items,
        threshold=SIMILARITY_THRESHOLD,
        use_preprocessing=USE_PREPROCESSING,
    )

    with open("duplicates.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
