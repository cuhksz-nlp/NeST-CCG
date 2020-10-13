unzip candc.zip

cd candc

make -f Makefile.unix clean
make -f Makefile.unix

make -f Makefile.unix all train bin/generate

./bin/generate
