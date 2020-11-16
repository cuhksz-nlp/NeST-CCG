#!/bin/bash

python data_processing.py --ccgbank_home=./LDC05T13 --supertag --ccg_parsing

cp ../sample_data/425tags ../data/

# You need to make sure ./candc/bin/generate can run appropriately before you run the following code
# The following code is used to generate the files to evaluate the CCG parsing results.
# You do not need to run the following code if you only focus on CCG supertagging and do not want to evaluate the ccg parsing results.

chmod +x *.sh

./create_data.sh ../data/tmp ../data/tmp/working ../candc

sed '1,3d' ../data/tmp/working/gold/wsj00.stagged > ../data/gold_files/dev.stagged
sed '1,3d' ../data/tmp/working/gold/wsj23.stagged > ../data/gold_files/test.stagged

# You can comment the following command line if you do not want to delete the tmp data
rm -rf ../data/tmp
