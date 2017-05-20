#!/usr/bin/env bash

python -m experiment \
  --save SemEval \
  --auxiliary-tasks none

python -m experiment \
  --save SemEval+ACE \
  --auxiliary-tasks ACE

python -m experiment \
  --save SemEval+Conll2000POS \
  --auxiliary-tasks Conll2000POS

python -m experiment \
  --save SemEval+Conll2000Chunk \
  --auxiliary-tasks Conll2000Chunk

python -m experiment \
  --save SemEval+GMB-NER \
  --auxiliary-tasks GMB-NER
