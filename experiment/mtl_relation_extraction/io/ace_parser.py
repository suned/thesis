import os
from _elementtree import ParseError
from xml.etree import ElementTree
import io
from nltk import sent_tokenize

import spacy

from ..ground_truth import GroundTruth, BadTokenizationError
from .. import log

nlp = spacy.load("en")


class NoSentenceFondException(Exception):
    pass


class NonBinaryRelationError(Exception):
    pass


def get_arguments(relation):
    all_args = []
    mentions = get_mentions(relation)
    for mention in mentions:
        arguments = get_mention_args(mention)
        if len(arguments) != 2:
            raise NonBinaryRelationError()
        arg1, arg2 = arguments
        role = get_role(arg1)
        arg1 = find_start(arg1), find_end(arg1)
        arg2 = find_start(arg2), find_end(arg2)
        all_args.append((role, arg1, arg2))
    return all_args


def get_mention_args(mention):
    entity_args = []
    arguments = mention.findall("./relation_mention_argument")
    for argument in arguments:
        if (get_role(argument) == "Arg-1"
            or get_role(argument) == "Arg-2"):
            entity_args.append(argument)
    return entity_args


def get_mentions(relation):
    return relation.findall("./relation_mention")


def find_end(arg):
    return int(arg.find("./extent/charseq").attrib["END"])


def get_role(arg):
    return arg.attrib["ROLE"]


def in_sentence(sentence, arg, text):
    arg_start, arg_end = arg
    sent_start = text.find(sentence)
    sent_end = sent_start + len(sentence)
    return arg_start >= sent_start and arg_end <= sent_end


def find_sentence(text, arg1, arg2):
    arg1_start, arg1_end = arg1
    arg2_start, arg2_end = arg2
    sents = sent_tokenize(text)
    for i, sent in enumerate(sents):
        sent_start = text.find(sent)
        if (in_sentence(sent, arg1, text)
            and in_sentence(sent, arg2, text)):
            arg1_in_sent, arg2_in_sent = get_in_sentence_indices(
                arg1_start,
                arg1_end,
                arg2_start,
                arg2_end,
                sent_start
            )
            return sent, arg1_in_sent, arg2_in_sent
    raise NoSentenceFondException()


def get_in_sentence_indices(arg1_start, arg1_end, arg2_start, arg2_end, sent_start):
    arg1_in_sentence = (arg1_start - sent_start,
                        arg1_end - sent_start + 1)
    arg2_in_sentence = (arg2_start - sent_start,
                        arg2_end - sent_start + 1)
    return (
        arg1_in_sentence,
        arg2_in_sentence
    )


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
    sgm_path = os.path.join(root, doc_id + ".sgm")
    try:
        text = get_text(sgm_path)
        relations = apf.findall(".//relation")
        for relation in relations:
            relation_id = relation.attrib["ID"]
            relation_type = relation.attrib["TYPE"]
            arguments = get_arguments(relation)
            for role, arg1, arg2 in arguments:
                try:
                    (sentence,
                     arg1_sent_idx,
                     arg2_sent_idx) = find_sentence(
                        text,
                        arg1,
                        arg2
                    )
                    all_relations.append(
                        make_relation(
                            sentence,
                            arg1_sent_idx,
                            arg2_sent_idx,
                            relation_type,
                            role
                        )
                    )
                except NoSentenceFondException:
                    log.error(
                        "No sentence found in ACE doc %s, relation %s",
                        doc_id,
                        relation_id
                    )
                except BadTokenizationError:
                    log.error(
                        "Bad tokenization detected in ACE doc %s, relation %s",
                        doc_id,
                        relation_id
                    )
    except NonBinaryRelationError:
        log.error(
            "Non binary relation in ACE doce %s, relation %s",
            doc_id,
            relation_id
        )
    except ParseError as e:
        log.error("Parse error in ACE doc %s", doc_id)
    return all_relations


def find_start(arg):
    return int(arg.find("./extent/charseq").attrib["START"])


def get_doc_id(apf):
    return apf.find(".//document").attrib['DOCID']


def get_text(sgm_path):
    with open(sgm_path) as f:
        xml = f.read()
        xml_escaped = xml.replace("&", "&amp;")
        text = (""
            .join(
                ElementTree.parse(
                    io.StringIO(xml_escaped)
                ).find(".").itertext()
            ).replace("\n", " "))
        return text


def read_files(path):
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
