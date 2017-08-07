#!/usr/bin/env bash

python -m experiment SemEval \
  --auxiliary-tasks none \
  --iterations 1 \
  --multi-channel \
  --learning-surface

python -m experiment SemEval+ACE \
  --auxiliary-tasks ACE \
  --iterations 1 \
  --multi-channel \
  --learning-surface

python -m experiment SemEval+Conll2000POS \
  --auxiliary-tasks Conll2000POS \
  --iterations 1 \
  --multi-channel \
  --learning-surface

python -m experiment SemEval+Conll2000Chunk \
  --auxiliary-tasks Conll2000Chunk \
  --iterations 1 \
  --multi-channel \
  --learning-surface

python -m experiment SemEval+GMB-NER \
  --auxiliary-tasks GMB-NER \
  --iterations 1 \
  --multi-channel \
  --learning-surface

python -m experiment SemEval+ACE_share_all_filters
  --auxiliary-tasks ACE \
  --iterations 1 \
  --share-filters \
  --learning-surface
