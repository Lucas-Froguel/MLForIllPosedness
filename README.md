# Problem

This repo is to test approaches to solving the ill-posedness problem. That is, given the following equation
g(k) = \int_0^\infty K(k, x)f(x) dx 
and given that we know K(k, x) and g(k), we want to find f(x). Our idea is that if we have many functions K_n(k, x), then we would be able to better learn the dynamics of f(x) without fitting some noise. 

# Data Generation

We chose a test function f(x) and five kernels K_i(k, x) and calculated g(k) for 10^8 (from 0 to 10k with steps of 0.0001) points for each one of them. This code in julia automatically saves them in .csv files. Then, we saved this data in mongodb, big it was so big to instatiate both this data and the neural net in memory (this code is `load_database.pi`). We also created an index to speed things up, with 
```
db.getCollection('data').createIndex({"kernel": 1, "idx": 1})
```

This first test uses noiseless data. 

I still have not solved the problem of training with data from all kernels as the same time. 
