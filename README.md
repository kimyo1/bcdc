bcdc
-------
Bayesian classifier for the dynamical state of clusters

Please cite Kim et al.,2026, submitted


Now we are working on it...

Install
-------

Base (runtime only):

1.Go to the folder  

    cd bcdc_v.1.0/bcdc #(folder including pyproject.toml)

2.Install it

    pip install -e .

3.Then check it is installed

    pip show bcdc
    
Running
-------
We have two modules inside of it

1.bcdc for 3 dynamical state classification with 6 indicators.

    from bcdc3 import bcdc3
    bcdc3(data)
    
- `data` should be a 6d array with indicators ordered as 'sparsity','m12', 'fsub', 'kuiperV', 'doff', 'asym'. (See details in the paper.)
  
**Results** return with python dictionary format. Now only the label can be returned.

2.bcdc for 3 dynamical state classification with projected classifier to lower dimensional indicator spaces.

    from bcdc3 import bcdc3proj
    bcdc3proj(numb,i, j, m, n, l, q, data, clid)

- `numb` is for projected dimension. 
- `i` to `q` means each column of indicators. If you use it to lower dimension, you should change q to m to 0. (5 dimension, q=0, 4 dimension, l=0, q=0).
- `data` is n dimensional array (n== number). The order should be matched with the classifier's.
- `clid` is cluster id or name.

**Results** return with python dictionary format. Now you can get the classified clusters' names and probabilities for each dynamical state. 

*You can check detailed usages in the example jupyter notebook file.*

    




