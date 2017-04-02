import spacy

nlp = spacy.load(
    "en_core_web_sm",
    parser=False,
    tagger=False,
    entity=False
)


def tokenize(s):
    return nlp.tokenizer(s)
