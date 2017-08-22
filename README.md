# K-means Clustering from Input csv (Python3.5)

"""
This is a pure Python implementation of the K-Means Clustering algorithm.
I started from the code [here](https://www.snip2code.com/Snippet/7977/A-pure-python-implementation-of-K-Means)-
which is itself built from the code [here](http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering)

There are a few major differences:
- data points can be labeled
- data is read from a file and can be filtered for particular entries on a given column
- results are output to a file
- data input is normalized before running the algorithm
- initial clusters can be given instead of randomly chosen from the set
- removed the integration with Plotly since reading from a file means larger dimensions
- original ran on python2.7, this version runs on 3.5
    - (while I kept back compatibility in early commits I can't garantee it all the way)

The main issue with the original code arises for data with a lot of similar data points
which can result in empty clusters during the optimization
-> this is solved by fixing the centroid where it is as long as there is no other point in the cluster


"""

## Input File Encoding
Files need to be in Unicode (watch out for UTF-8 caracters in your files otherwise)

## How it works
The inputs are all in the first part of the kmeans file; you just need to adapt them and run the script

The file clustering_data.csv contains the an example of input with real data from routes searched for on Skyscanner

If you want to run the script multiple times you can run the iterate_kmeans.py script, it will output a file for each run

## Problems?
Ping me
