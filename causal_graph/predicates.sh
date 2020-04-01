#!/bin/bash

file=$1

batchsize=100
i=0
start=1

lines="$(wc --lines $file | egrep -o '[0-9]*\s')"

find . -name predicates.xml -exec rm -i {} \;
mkdir .out

while true
do
    i=$(($i+1))
    end=$(($start+$batchsize-1))
    sed -n $start,$((end))p $file > .text$i.tmp
    echo semrep.v1.8 -X .text$i.tmp .out/out$i".xml" >> commands.txt
    if [ $end -ge $lines ]; then
	break
    fi
    start=$(($end+1))
done

parallel < commands.txt

echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" > predicates.xml
echo "<!DOCTYPE SemRepAnnotation PUBLIC \"-//NLM//DTD SemRep Output//EN\" \"http://semrep.nlm.nih.gov/DTD/SemRepXML_v1.8.dtd\">" >> predicates.xml
echo "<root>" >> predicates.xml

clear
echo "######### FIXING SEMREP'S CORE DUMPED #####################"

for i in .out/*                               #fixing .xmls that semrep core dumped
do
    xmllint --noout $i
    valid=$?
    c=0
    while [[ valid -ne 0 ]]                   #looping till semrep works
    do
	n="$(echo $i | egrep -o "[0-9]+")"
	semrep.v1.8 -X .text$n.tmp $i
	xmllint --noout $i
	valid=$?
	c=$(($c+1))
	if [ $c -gt 10 ];then
	    break
	fi
    done
    if [ $valid -eq 0 ];then
	sed 1,2d $i >> predicates.xml
    fi
done

echo "</root>" >> predicates.xml


rm commands.txt
rm .text*
rm -r .out/

