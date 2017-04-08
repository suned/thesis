import os
from _elementtree import ParseError
from xml.etree import ElementTree


import spacy

from ..ground_truth import GroundTruth, BadTokenizationError
from .. import log
nlp = spacy.load("en")


class NoSentenceFondException(Exception):
    pass


def get_arguments(relation):
    all_args = []
    mentions = relation.findall("./relation_mention")
    for mention in mentions:
        arguments = mention.findall("./relation_mention_argument")
        arg1, arg2 = arguments
        role = get_role(arg1)
        arg1 = find_start(arg1), find_end(arg1)
        arg2 = find_start(arg2), find_end(arg2)
        all_args.append((role, arg1, arg2))
    return all_args


def find_end(arg):
    return int(arg.find("./extent/charseq").attrib["END"])


def get_role(arg):
    return arg.attrib["ROLE"]


def find_sentence(text, arg1, arg2):
    arg1_start, arg1_end = arg1
    arg2_start, arg2_end = arg2
    doc = nlp(text)
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        if (arg1_start + 1 > sent_start and
            arg1_end - 1 < sent_end and
            arg2_start > sent_start and
            arg2_end < sent_end):
            arg1_in_sentence = (arg1_start - sent_start,
                                arg1_end - sent_start + 1)
            arg2_in_sentence = (arg2_start - sent_start,
                                arg2_end - sent_start + 1)
            return (
                sent.text.replace("\n", " "),
                arg1_in_sentence,
                arg2_in_sentence
            )
    raise NoSentenceFondException()


def make_relation(sentence, arg1, arg2, relation_type, role):
    if role == "Arg-1":
        arguments = ("e1", "e2")
    else:
        arguments = ("e2", "e1")
    return GroundTruth(
        sentence_id=None,
        sentence=sentence,
        e1_offset=arg1,
        e2_offset=arg2,
        relation=relation_type,
        arguments=arguments
    )


def get_relations(apf_file):
    all_relations = []
    root, apf = apf_file
    doc_id = get_doc_id(apf)
    log.info("Parsing ACE file %s", doc_id)
    sgm_path = os.path.join(root, doc_id + ".sgm")
    try:
        text = get_text(sgm_path)
        relations = apf.findall(".//relation")
        for relation in relations:
            relation_type = relation.attrib["TYPE"]
            arguments = get_arguments(relation)
            for role, arg1, arg2 in arguments:
                try:
                    sentence, arg1, arg2 = find_sentence(text, arg1, arg2)
                    all_relations.append(
                        make_relation(sentence, arg1, arg2, relation_type, role)
                    )
                except NoSentenceFondException:
                    relation_id = relation.attrib["ID"]
                    log.error(
                        "No sentence found in doc %s, relation %s",
                        doc_id,
                        relation_id
                    )
                except BadTokenizationError:
                    relation_id = relation.attrib["ID"]
                    log.error(
                        "Bad tokenization detected in %s, relation %s",
                        doc_id,
                        relation_id
                    )
    except ParseError:
        log.error("Parse error in %s", doc_id)
    return all_relations


def find_start(arg):
    return int(arg.find("./extent/charseq").attrib["START"])


def get_doc_id(apf):
    return apf.find(".//document").attrib['DOCID']


def get_text(sgm_path):
    return "".join(ElementTree.parse(sgm_path).find(".").itertext())


def parse(path):
    relations = []
    apf_files = read_apf_xml(path)
    for apf_file in apf_files:
        relations.extend(get_relations(apf_file))
    return relations


def read_apf_xml(path):
    trees = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".apf.xml") and "adj" in root:
                file_path = os.path.join(root, filename)
                tree = ElementTree.parse(file_path)
                trees.append((root, tree))
    return trees