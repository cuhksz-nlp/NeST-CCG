# The following is modified from candc/src/scripts/ccg/create_data.sh

CCGBANK=$1
WORK=$2
CANDC=$3

BIN=bin
SCRIPTS=$CANDC/src/scripts/ccg
DATA=src/data/ccg

GOLD=$WORK/gold
GEN=$WORK/generated
LOG=$WORK/log
MARKEDUP=$DATA/cats/markedup

mkdir -p $WORK

mkdir -p $GOLD
mkdir -p $GEN
cat /dev/null > $LOG

echo 'converting AUTO files to old .pipe format' | tee -a $LOG

# For evaluation, we only need dev and test
$SCRIPTS/convert_auto.sh $CCGBANK/dev.auto | ./convert_brackets.sh > $GOLD/wsj00.pipe

$SCRIPTS/convert_auto.sh $CCGBANK/test.auto | ./convert_brackets.sh > $GOLD/wsj23.pipe

echo 'extracting CCGbank POS/super-tagged text' | tee -a $LOG

python2 $SCRIPTS/extract_sequences.py -s $GOLD/wsj00.pipe > $GOLD/wsj00.stagged
python2 $SCRIPTS/extract_sequences.py -s $GOLD/wsj23.pipe > $GOLD/wsj23.stagged
