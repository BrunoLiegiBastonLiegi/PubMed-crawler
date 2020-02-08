#!/bin/bash

query=$1
curl -X DELETE "localhost:9200/$query?pretty" 

batchsize=10000
lines="$(wc --lines json/$query/$query.json | egrep -o '[0-9]*')"

i=0
start=1
while true
do
    i=$(($i+1))
    end=$(($start+2*$batchsize-1))
    filename="json/"$query"/"$query"_part_"$i".json"
    sed -n $start,$((end))p json/$query/$query.json > $filename
    curl -H "Content-Type: application/json" -XPOST "localhost:9200/$query/_bulk?pretty&refresh" --data-binary ""@"$filename" 
    if [ $end -ge $lines ]; then
	break
    fi
    start=$(($end+1))
done

curl "localhost:9200/_cat/indices?v"
