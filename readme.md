# bakalaureus

This repositorium was created to manage the files used in Kristjan Poska's bachelor's thesis.

The protocol texts originate from the crowdsourcing project of [The National Archives of Estonia](https://www.ra.ee/vallakohtud/), and the manual annotations have been created in the project "Possibilities of automatic analysis of historical texts by the example of 19th-century Estonian communal court minutes". The project "Possibilities of automatic analysis of historical texts by the example of 19th-century Estonian communal court minutes" is funded by the national programme "Estonian Language and Culture in the Digital Age 2019-2027". 

# Thesis topic: 
Named Entity Recognition in 19th Century Parish Court Protocols

# Thesis instructior:
Siim Orasmaa, PhD

# Libraries and technologies used in the thesis:

### Technologies:
- Python 3.7
- [EstNLTK](https://github.com/estnltk/estnltk/tree/master) (version 1.6). The files from the [estner](https://github.com/estnltk/estnltk/tree/devel_1.6/estnltk/taggers/estner) folder of branch [devel_1.6](https://github.com/estnltk/estnltk/tree/devel_1.6) were also used (state of files at commit [ebf1451](https://github.com/estnltk/estnltk/commit/ebf1451e69a2327502021e50571e318af1852ab2#diff-ebc8422f5e537e04a286bc2df0c9c830311f142c0494a47d42878097723d26ea)).
- Jupyter Notebooks (+ Anaconda)

### Libraries/dependencies:
- [nervaluate 0.1.8](https://pypi.org/project/nervaluate/)
- [pandas 1.2.0](https://pandas.pydata.org/)
- os, json, random, sys, re
