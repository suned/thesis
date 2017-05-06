rsync \
  -ra \
  --progress \
  --exclude=".*" --exclude="__pycache__" \
  data \
  experiment \
  results \
  GloVe \
  nkz509@ssh-diku-apl.science.ku.dk:/home/nkz509/hdrive/thesis
