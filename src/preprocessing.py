import re
from collections import Counter

def tokenize(text):
    def clean_text(text2):
        text2 = text2.lower()
        text2 = re.sub(r"https?://\S+|www\.\S+", " url ", text2)
        text2 = re.sub(r"@\w+", " user ", text2)
        text2 = re.sub(r"#", "", text2)
        text2 = re.sub(r"[^a-z0-9\s']", " ", text2)
        text2 = re.sub(r"\s+", " ", text2).strip()
        return text2

    return clean_text(text).split()

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(texts, min_freq=2, max_vocab_size=20000):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def numericalize(text, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]

def add_keyword_to_input_text(df):
    keyword = (
        df["keyword"]
        .fillna("")
        .str.replace("%20", " ", regex=False)
        .str.strip()
    )

    input_text = df["input_text"].fillna("").str.strip()

    return (keyword + " " + input_text).str.strip()