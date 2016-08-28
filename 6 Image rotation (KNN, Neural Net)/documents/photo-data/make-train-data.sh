#!/bin/bash

cat $1 | while read p; do
   fname=`echo $p | awk '{print $1;}'`
   label=`echo $p | awk '{print $2;}'`
   for i in 0 90 180 270; do
       echo -n "$fname $i "
       convert -resize 8x8! -rotate $i $fname txt:- | grep 'rgb(' | sed 's/.*rgb.\([0-9]*\),\([0-9]*\),\([0-9]*\)./\1 \2 \3/g;' | xargs echo -n
       echo
   done
done | awk 'NF == 194 { print $0; }'

