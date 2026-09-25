"""
Language detection starter.
"""

import sys
from pathlib import Path

# pylint: disable=unused-variable, duplicate-code

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab_1_classify_profile.main import (
    calculate_frequencies,
    calculate_mse,
    compare_profiles_by_mse,
    compare_profiles_by_top_n,
    create_language_profile,
    detect_language_by_mse,
    detect_language_by_top_n,
    get_top_n_words,
    remove_stop_words,
    tokenize,
)

def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    result = None


    de_tokens = tokenize(de_text)
    de_tokens = remove_stop_words(de_tokens, stopwords)
    de_frequencies = calculate_frequencies(de_tokens)
    top_7_words = get_top_n_words(de_frequencies, 7)

    print(top_7_words)

    en_profile = create_language_profile(
        "en",
        en_text,
        stopwords
    )

    de_profile = create_language_profile(
        "de",
        de_text,
        stopwords
    )

    unknown_profile = create_language_profile(
        "unknown",
        unknown_text,
        stopwords
    )

    result = detect_language_by_top_n(
        unknown_profile,
        en_profile,
        de_profile,
        15
    )

    mse_to_en = compare_profiles_by_mse(unknown_profile, en_profile)
    mse_to_de = compare_profiles_by_mse(unknown_profile, de_profile)
    result_by_mse = detect_language_by_mse(unknown_profile, en_profile, de_profile)

    print("Top-n result:", result)
    print("MSE to en:", mse_to_en)
    print("MSE to de:", mse_to_de)
    print("MSE result:", result_by_mse)

    assert result, "Detection result is None"
    assert result_by_mse, "MSE detection result is None"


if __name__ == "__main__":
    main()
