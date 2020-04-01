PubMed Crawler


Python project for retrieval and processing of medical articles from the PubMed database. 
The script crawler.py performs a query to the database and from each article found collects: abstract, title, author, date, journal and keywords list. The results then are stored both in xml format, and json format for Elasticsearch indexing.
preprocessor.py implements a Preprocessor object that is used to load data from the xml, or simply to preprocess raw text. It splits the text in paragraphs and sentences and then performs tokenization (with regex), stop words removal and lemmatization (with python nltk).

In causal_graph/ I use SemRep (https://github.com/lhncbc/SemRep) to extract the predications and then build the causal graph using python graph_tool.
