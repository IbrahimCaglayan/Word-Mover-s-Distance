# Word-Mover-s-Distance

The project consists in writing and comparing different methods of distance calculation between documents. Distances calculations’ methods are described in the fourth section of the following article.
Kusner et al., 2015 - http://proceedings.mlr.press/v37/kusnerb15.pdf

The written methods are the following :
- Word centroid distance (WCD) between 2 documents
- Relaxed word moving distance (RWMD) between 2 documents
- Exact WMD via Prefetch and prune to find the k-nearest neighbors of an input sentence within a collection of reference sentences.

WCD and RWMD are tested in the following sentences : "Obama speaks to the media in Illinois" and "The President greets the press in Chicago." 
In order to compare other sentences, the chosen ones can be changed.

Each function’s time for execution os displayed in the code
