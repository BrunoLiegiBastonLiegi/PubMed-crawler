# PubMed Crawler

Python project for retrieval and processing of medical articles from the PubMed database.

## Retrieving articles and storing locally
The script `crawler.py` uses the [Entrez utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) to perform a query to the PubMed database, with query keyword defined inside. For each article found is allocated an `Article` object, defined in 'article.py', that collects: abstract, title, author, date, journal and keywords list. The results then are stored both in xml format (`DATA/xml/`), and json format (`elasticsearch/json/`) for Elasticsearch indexing. The task is parallelized on the number of available threads using python `multiprocessing` module.

### Requirements
```
requests
Beautifulsoup
multiprocessing
```

## Preprocessing 

`preprocessor.py` implements a `Preprocessor` object that is used to load data from the xml generated at the previous step, or simply to preprocess raw text. It splits the text in paragraphs and sentences and then performs tokenization (with regex), stop words removal and lemmatization (with nltk). Again `multiprocessing` is used to parallelize the task on the available threads.

### Requirements

```
re
Beautifulsoup
multiprocessing
nltk
```

## Causal graph

Here I use [SemRep](https://github.com/lhncbc/SemRep) to extract the predications from the abstracts collected with `crawler.py`. The bash script `predicates.sh` takes as argument a raw text file with one sentence per line (can be generated using `Preprocessor` for example)
```
sh predicates.sh example.txt
```
and generates the `predicates.xml` file containing all the predications found by SemRep. The task is splitted in batches of 100 sentences and runs parallel thanks to [GNU Parallel](https://www.gnu.org/software/parallel/).
`causal_graph.py` then parses the predications file generated and extracts all the subject-predicate-object tuples. From them selects only the causal one and build the graph using the `Graph` object implemented in `graph.py` and based on the python module [graph_tool](https://graph-tool.skewed.de/).

### Requirements

```
SemRep
GNU Parallel
graph_tool
Beautifulsoup
```