
for i in .out/*                               #fixing .xmls that semrep core dumped
do
    xmllint --noout $i
    valid=$?
    if [ $valid -eq 0 ];then
	sed 1,2d $i >> predicates-tmp.xml
    fi
done
