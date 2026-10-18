"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable


from main import collect_frequencies, decode, get_vocabulary, train


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()

    word_frequencies = collect_frequencies(text, start_of_word=None, end_of_word="</s>")
    assert word_frequencies is not None, "Word frequencies should be built successfully"

    trained_tokens = train(word_frequencies, 100)
    assert trained_tokens is not None, "BPE training should produce a vocabulary"

    vocabulary = get_vocabulary(trained_tokens, unknown_token="<unk>")
    assert vocabulary is not None, "Token vocabulary should be built successfully"

    with open(
        "lab_2_tokenize_by_bpe/assets/secrets/secret_1.txt", "r", encoding="utf-8"
    ) as secret_file:
        encoded_secret = [int(token) for token in secret_file.read().split()]
    decoded_secret = decode(encoded_secret, vocabulary, end_of_word_token="</s>")
    assert decoded_secret is not None, "Secret text should be decoded successfully"

    merged_tokens = sorted(
        {token for word in trained_tokens for token in word if len(token) > 1}
    )
    print(f"Total vocabulary tokens: {len(vocabulary)}")
    print(f"Merged tokens found: {len(merged_tokens)}")
    print("First merged tokens:")
    for token in merged_tokens[:10]:
        print(token)
    print("Decoded secret:")
    print(decoded_secret)


if __name__ == "__main__":
    main()
