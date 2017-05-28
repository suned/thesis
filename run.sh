#!/usr/bin/env bash

python -m experiment \
  --save SemEval \
  --auxiliary-tasks none \
  --share-filters

python -m experiment \
  --save SemEval+ACE \
  --auxiliary-tasks ACE \
  --share-filters

python -m experiment \
  --save SemEval+Conll2000POS \
  --auxiliary-tasks Conll2000POS \
  --share-filters

python -m experiment \
  --save SemEval+Conll2000Chunk \
  --auxiliary-tasks Conll2000Chunk \
  --share-filters

python -m experiment \
  --save SemEval+GMB-NER \
  --auxiliary-tasks GMB-NER \
  --share-filters
