import os
import numpy

from ..mtl_relation_extraction.ground_truth import Sequence

sentences = None


def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename),
                          'rb') as file_handle:
                    file_content = file_handle.read().decode(
                        'utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in
                                            annotated_sentence.split(
                                                '\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(
                                annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], \
                                             annotations[1], \
                                             annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in (
                            'LQU', 'RQU'):  # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append(
                                (word, tag, ner))

                        yield to_conll_iob(
                            standard_form_tokens)


def get_words(sentence):
    return [token[0] for token in sentence]


def get_entities(sentence):
    return [token[2] for token in sentence]


def gmb_named_entities(corpus_root):
    lazy_load_sentences(corpus_root)
    sequences = []
    for sentence in sentences:
        words = get_words(sentence)
        tags = get_entities(sentence)
        sequence = Sequence(
            words,
            tags
        )
        sequences.append(sequence)
    return numpy.array(sequences)


def lazy_load_sentences(corpus_root):
    global sentences
    if sentences is None:
        sentences = read_gmb(corpus_root)
