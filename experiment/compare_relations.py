from json import JSONDecodeError

import nltk
from stanfordcorenlp import StanfordCoreNLP
import os

from matplotlib import pyplot
from .plot.util import figure_size
from matplotlib_venn import venn3

from .io import arguments
arguments.n_grams = []
arguments.word_embedding_dimension = 300
arguments.data_path = "data/"
from .mtl_relation_extraction.ace_task import ACE
from .mtl_relation_extraction.semeval_task import SemEvalTask
from .mtl_relation_extraction import log

nlp = StanfordCoreNLP("http://localhost")


class NotNounPhraseError(Exception):
    pass


class BadEntityIndexError(Exception):
    pass


def arguments_are_pronouns_set(task):
    pronouns_set = set()
    for relation in task.relations:
        tokens = [str(token) for token in relation.sentence]
        pos_tags = [tag for _, tag in nltk.pos_tag(tokens, tagset="universal")]
        e1_tags = pos_tags[slice(*relation.e1)]
        e2_tags = pos_tags[slice(*relation.e2)]
        if "PRON" in e1_tags or "PRON" in e2_tags:
            pronouns_set.add(relation.sentence_id)
    return pronouns_set


def find_parent_noun_phrase(tree, entity):
    entity_start, _ = entity
    try:
        position = tree.leaf_treeposition(entity_start)
        depth = len(position) - 1
        while depth != 0 and "NP" not in tree[position[:depth]].label():
            depth -= 1
        if depth == 0:
            raise NotNounPhraseError
        return tree[position[:depth]].treeposition()
    except IndexError:
        raise BadEntityIndexError


def plot_syntactic_analysis(path):
    ace = ACE()
    semeval = SemEvalTask()

    ace.load()
    semeval.load()

    for i, relation in enumerate(ace.relations):
        relation.sentence_id = i

    ace_relations = set(relation.sentence_id for relation in ace.relations)
    semeval_relations = set(relation.sentence_id for relation in semeval.relations)

    ace_same_np_set = count_same_np(ace)
    semeval_same_np_set = count_same_np(semeval)

    ace_pronouns_set = arguments_are_pronouns_set(ace)
    semeval_pronouns_set = arguments_are_pronouns_set(semeval)
    figure_path = os.path.join(path, "ace_syntactic_analysis.pgf")
    venn(ace_relations, ace_same_np_set, ace_pronouns_set, "ACE 2005", figure_path)
    figure_path = os.path.join(path, "semeval_syntactic_analysis.pgf")
    venn(semeval_relations, semeval_same_np_set, semeval_pronouns_set, "SemEval 2010 Task 8", figure_path)


def venn(relations, nps, pronouns, task_name, path):
    pyplot.figure(figsize=figure_size())
    venn3(
        [relations, nps, pronouns],
        [task_name, r"$sameNP(arg1, arg2)$", r"$pronoun(arg1, arg2)$"]
    )
    pyplot.tight_layout()
    pyplot.savefig(path)


def count_same_np(task):
    same_np_set = set()
    for relation in task.relations:
        try:
            parse = nlp.parse(relation.sentence.text)
            tree = nltk.ParentedTree.fromstring(parse)

            e1_noun_phrase_position = find_parent_noun_phrase(tree, relation.e1)
            e2_noun_phrase_position = find_parent_noun_phrase(tree, relation.e2)
            if e1_noun_phrase_position == e2_noun_phrase_position:
                same_np_set.add(relation.sentence_id)
        except NotNounPhraseError:
            log.error("No parent noun phrase found for argument")
        except BadEntityIndexError:
            log.error("Entity was wrongly indexed")
        except JSONDecodeError:
            log.error("CoreNLP timeout")
    log.info("Finished task %s", task.name)
    return same_np_set





