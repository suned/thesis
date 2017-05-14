#!/usr/bin/env bash

python -m experiment --save SemEval --auxiliary-tasks none
python -m experiment --save SemEval+ACE --auxiliary-tasks ACE
