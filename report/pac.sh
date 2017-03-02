CHARS=$(detex sune_debel_master_thesis.tex | wc -m)
CHARS_PR_PAGE=2400
echo "$(($CHARS / $CHARS_PR_PAGE))"
