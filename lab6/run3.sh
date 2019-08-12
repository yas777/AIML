./encoder.sh $1 $3 > mymdp.txt
./valueiteration.sh mymdp.txt > myval.txt
./decoder.sh $1 myval.txt $3 > $2