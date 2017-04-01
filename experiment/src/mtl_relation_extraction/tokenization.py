import spacy

nlp = spacy.load("en_core_web_sm")


def tokenize(s):
    return nlp.tokenizer(s)
