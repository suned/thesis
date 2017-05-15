#!/usr/bin/env bash

rm -r results/*

python -m experiment \
  --save SemEval \
  --auxiliary-tasks none

python -m experiment \
  --save SemEval+ACE \
  --auxiliary-tasks ACE

python -m experiment \
  --save SemEval+ACE_sequential \
  --auxiliary-tasks ACE \
  --fit-sequential

python -m experiment \
  --save SemEval+Conll2000POS \
  --auxiliary-tasks Conll2000POS

python -m experiment \
  --save SemEval+Conll2000POS_sequential \
  --auxiliary-tasks Conll2000POS \
  --fit-sequential

python -m experiment \
  --save SemEval+Conll2000Chunk \
  --auxiliary-tasks Conll2000Chunk

python -m experiment \
  --save SemEval+Conll2000Chunk_sequential \
  --auxiliary-tasks Conll2000Chunk \
  --fit-sequential

python -m experiment \
  --save SemEval+GMB-NER \
  --auxiliary-tasks GMB-NER

python -m experiment \
  --save SemEval+GMB-NER_sequential \
  --auxiliary-tasks GMB-NER \
  --fit-sequential
