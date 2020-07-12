#!/bin/bash

SCRIPTS=$1/src/scripts/ccg
DATA=$1/src/data/ccg

MARKEDUP=$DATA/cats/markedup

$SCRIPTS/convert_auto.sh $2 | sed -f $SCRIPTS/convert_brackets.sh > tmp.pipe

echo 'bin/generate -j produces CCGbank slot dependencies,'

$1/bin/generate -j $DATA/cats $MARKEDUP tmp.pipe > $3

rm -rf tmp.pipe
sed -i -e 's/^$/<c>n/' $3
rm -rf $3-e
