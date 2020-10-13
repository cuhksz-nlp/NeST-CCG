# this file is modified from candc/src/scripts/ccg/convert_brackets.sh

sed 's/-LRB-/(/g;' | sed 's/-RRB-/)/g;' | sed 's/-LCB-/{/g;' | sed 's/-RCB-/}/g;' | sed 's/-LSB-/[/g;' | sed 's/-RSB-/]/g;'

