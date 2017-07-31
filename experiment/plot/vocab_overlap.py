from matplotlib_venn import venn2
from matplotlib import pyplot
import os

from ..io import arguments

arguments.data_path = "/Users/sune/ITAndCognition/thesis/data"
arguments.word_embedding_dimension = 300
arguments.n_grams = [1]
from ..mtl_relation_extraction.semeval_task import SemEvalTask
from ..mtl_relation_extraction.ace_task import ACE
from ..mtl_relation_extraction.conll2000_chunk_task import Conll2000ChunkTask
from ..mtl_relation_extraction.conll2000_pos_task import Conll2000PosTask
from ..mtl_relation_extraction.gmb_ner_task import GMBNERTask
from .util import figure_size


def venn(vocab1, vocab2, vocab1_label, vocab2_label, path):
    pyplot.figure(figsize=figure_size())
    venn2([vocab1, vocab2], [vocab1_label, vocab2_label])
    pyplot.savefig(path)


def plot_venn_diagrams(path):
    semeval = SemEvalTask()
    ace = ACE()
    conll_pos = Conll2000PosTask()
    gmb = GMBNERTask()

    semeval.load()
    ace.load()
    conll_pos.load()
    gmb.load()

    semeval_vocab = semeval.get_vocabulary()
    ace_vocab = ace.get_vocabulary()
    conll_pos_vocab = conll_pos.get_vocabulary()
    gmb_vocab = gmb.get_vocabulary()

    semeval_label = "SemEval 2010 Task 8"
    ace_label = "ACE 2005"
    conll_pos_label = "CONLL2000"
    gmb_label = "Groningen Meaning Bank"

    ace_path = os.path.join(path, "ace_vocab_overlap.pgf")
    conll_pos_path = os.path.join(path, "conll_pos_vocab_overlap.pgf")
    gmb_path = os.path.join(path, "gmb_vocab_overlap.pgf")

    venn(semeval_vocab, ace_vocab, semeval_label, ace_label, ace_path)
    venn(
        semeval_vocab,
        conll_pos_vocab,
        semeval_label,
        conll_pos_label,
        conll_pos_path
    )
    venn(
        semeval_vocab,
        gmb_vocab,
        semeval_label,
        gmb_label,
        gmb_path
    )
