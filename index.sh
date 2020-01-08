#!/bin/bash

batchsize=1000
lines="$(wc --lines json/articles.json | egrep -o '[0-9]*')"

i=0
start=1
while [ $start -le $lines ]
do
    i=$(($i+1))
    end=$(($start+2*$batchsize-1))
    filename="json/articles_part_"$i".json"
    sed -n $start,$((end))p json/articles.json > $filename
    curl -H "Content-Type: application/json" -XPOST "localhost:9200/pubmed/_bulk?pretty&refresh" --data-binary ""@"$filename"  
    start=$(($end+1))
done

    #curl -H "Content-Type: application/json" -XPOST "localhost:9200/pubmed/_bulk?pretty&refresh" --data-binary "@json/articles_part_"$k.json"

curl "localhost:9200/_cat/indices?v"
