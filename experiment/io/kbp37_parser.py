from funcparserlib import lexer, parser
import numpy
from typing import (
    List
)
from ..mtl_relation_extraction import log
from ..mtl_relation_extraction.ground_truth import GroundTruth, BadTokenizationError


def tokenize(s: str) -> List[lexer.Token]:
    def choice(choices):
        return r"|".join(choices)

    relation_types = [
        r"alternate_names",
        r"origin",
        r"spouse",
        r"title",
        r"employee_of",
        r"countries_of_residence",
        r"stateorprovinces_of_residence",
        r"cities_of_residence",
        r"country_of_birth",
        r"no_relation",
        r"subsidiaries",
        r"top_members/employees",
        r"founded(_by)?",
        r"country_of_headquarters",
        r"stateorprovince_of_headquarters",
        r"city_of_headquarters",
        r"members"
    ]
    specs = [
        ("Comma", (r",",)),
        ("Person", (r"per",)),
        ("Org", (r"org",)),
        ("Colon", (r":",)),
        ("Tab", (r'\t',)),
        ("OpenEndTag", (r"</",)),
        ("OpenStartTag", (r"<",)),
        ("CloseTag", (r">",)),
        ("EntityId", (r"e(1|2)",)),
        ("LPar", (r"\(",)),
        ("RPar", (r"\)",)),
        ('Number', (r'[0-9]+',)),
        ("RelationType", (choice(relation_types),)),
        ("Comment", (r"Comment:.*",)),
        ('DoubleQuote', (r'"',)),
        ('Text', (r'''[\w0-9 ,:;\(\)\$%'\-_=&\?!@~#*\+\./"\|\[\]`]+''',)),
        ("LineBreak", (r"\n",)),
    ]
    useless = []
    tokenizer = lexer.make_tokenizer(specs)
    return [x for x in tokenizer(s)
            if x.type not in useless]


def is_entity(part):
    return (part is not None
            and type(part) is tuple
            and (part[0] == "e1"
                 or part[0] == "e2"))


def make_sentence_string(sent):
    sentence = ""
    for part in sent:
        if part is not None:
            if is_entity(part):
                entity_id, text = part
                sentence += text.strip()
            else:
                sentence += part
    return sentence[1:-1]


def get_entities(sent):
    return (part for part in sent if is_entity(part))


def sentence_to_offset(sent):
    (e1, e1_text), (e2, e2_text) = get_entities(sent)
    e1_text = e1_text.strip()
    e2_text = e2_text.strip()
    first_part = sent[0]
    second_part = sent[2]
    third_part = sent[4]
    if first_part is None:
        e1_start = 0
    else:
        first_part = first_part[1:]
        e1_start = len(first_part)
    e1_end = e1_start + len(e1_text)
    if second_part is None:
        e2_start = e1_end + 1
    else:
        e2_start = e1_end + len(second_part)
    if third_part is None:
        e2_end = e2_start + len(e2_text) - 1
    else:
        e2_end = e2_start + len(e2_text)
    return (e1_start, e1_end), (e2_start, e2_end)


def parse(s: str) -> List[GroundTruth]:
    def tokval(x):
        return x.value

    def toktype(t):
        return parser.some(lambda x: x.type == t) >> tokval

    def join(strings):
        return "".join(strings)

    def make_relation(rel):
        sent_id, sent, rel_type, args = rel
        sent_string = make_sentence_string(sent)
        e1_offset, e2_offset = sentence_to_offset(sent)
        try:
            return GroundTruth(
                sent_id,
                sent_string,
                e1_offset,
                e2_offset,
                rel_type,
                args
            )
        except BadTokenizationError as e:
            log.error("Tokenization error detected in sentence: " + str(sent_id))
            return None

    number = toktype("Number") >> int
    arg_type = toktype("Person") | toktype("Org")
    colon = parser.skip(toktype("Colon"))
    sentence_id = number
    newline = parser.skip(toktype("LineBreak"))
    relation_type = toktype("RelationType") >> str
    text = toktype("Text") >> str
    open_start_tag = parser.skip(toktype("OpenStartTag"))
    tab = parser.skip(toktype("Tab"))
    close_tag = parser.skip(toktype("CloseTag"))
    lpar = parser.skip(toktype("LPar"))
    rpar = toktype("RPar") >> str
    entity_id = toktype("EntityId") >> str
    open_end_tag = parser.skip(toktype("OpenEndTag"))
    eof = parser.skip(parser.finished)
    comma = toktype("Comma") >> str
    text_but_first = lambda p: p + text >> join
    after_entity = (
        text | text_but_first(comma) | text_but_first(rpar)
    )

    entity_tag = (
        open_start_tag
        + entity_id
        + close_tag
    ) >> str
    entity_end_tag = parser.skip(
        open_end_tag
        + entity_id
        + close_tag
    )
    entity = (
        entity_tag
        + text
        + entity_end_tag
    ) >> tuple
    sentence = (
        parser.maybe(text)
        + entity
        + parser.maybe(after_entity)
        + entity
        + parser.maybe(after_entity)
    ) >> tuple
    relation_args = (
        parser.skip(lpar)
        + entity_id
        + parser.skip(comma)
        + entity_id
        + parser.skip(rpar)
    ) >> tuple
    relation = (
        sentence_id
        + tab
        + sentence
        + newline
        + parser.skip(parser.maybe(arg_type + colon))
        + relation_type
        + parser.maybe(relation_args)
        + newline
        + newline
        + parser.skip(parser.maybe(newline))
    ) >> make_relation
    relations = parser.many(relation)
    semeval = relations + eof

    tokens = tokenize(s)
    relations = semeval.parse(tokens)
    return numpy.array(
        [relation for relation in relations
         if relation is not None]
    )


def read_file(path: str) -> List[GroundTruth]:
    with open(path) as f:
        s = f.read()
    return parse(s)

