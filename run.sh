#!/usr/bin/env bash

python -m experiment SemEval \
  --auxiliary-tasks none \
  --iterations 5 \
  --learning-surface

python -m experiment SemEval+ACE \
  --auxiliary-tasks ACE \
  --iterations 5 \
  --learning-surface

python -m experiment SemEval+Conll2000POS \
  --auxiliary-tasks Conll2000POS \
  --iterations 5 \
  --learning-surface

python -m experiment SemEval+Conll2000Chunk \
  --auxiliary-tasks Conll2000Chunk \
  --iterations 5 \
  --learning-surface

python -m experiment SemEval+GMB-NER \
  --auxiliary-tasks GMB-NER \
  --iterations 5 \
  --learning-surface
