def split_sentences(document: str) -> list[str]:
    return [sentence.strip() for sentence in document.split('.')]
