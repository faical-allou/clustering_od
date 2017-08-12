#############################################################################
# Full Imports

import sys
import math
import random
import subprocess

"""
This is a pure Python implementation of the K-Means Clustering algorithmn.
I started from the code here:
https://www.snip2code.com/Snippet/7977/A-pure-python-implementation-of-K-Means-
which is itself built from the code here:
http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering

There are a few major differences:
- data points can be labeled
- data is read from a file and can be filtered for particular entries on a given column
- results are output to a file
- data input is normalized before running the algorithm
- initial clusters can be given instead of randomly chosen from the set
- removed the integration with Plotly since reading from a file means larger dimensions
- original ran on python2.7, this version runs on both 2.7 and 3.5

The main issue with the original code arises for data with a lot of similar data points
which can result in empty clusters during the optimization
-> this is solved by fixing the centroid where it is as long as there is no other point in the cluster

Files need to be in Unicode (watch out for UTF-8 caracters in your files otherwise)
"""

def main():
    # file name containing the data (first column should have column names)
    filename_input = 'C:/Users/faicalallou/Documents/Dev/od_clusters.csv'

    # file name of the initial centroids if any (use None to start from a random set)
    # use the same column name and structure as filename_input
    filename_init = 'C:/Users/faicalallou/Documents/Dev/initial_centroids.csv'
    #filename_init = None

    # file name to export to
    filename_out = 'C:/Users/faicalallou/Documents/Dev/clustering_results_export_test.csv'

    # indeces of the colum with ID (such as origin or category) in the input file
    # Can have multiple column for name ID.
    # then repeats the Centroid names to match record names during export
    id_column = [0,1,2]
    centroid_name = ''.join('Centroid,' for i in range(len(id_column)))[:-1]

    # index of columns to read from the file (index start at 0)
    first_column = 3
    last_column = 30
    list_index = range(first_column,last_column+1)

    # filtering the input file based on ID:
    # Note: the filter is done on the id_column and the index to filter refers to it
    # (i.e. "index_to_filter_on = 1 means the second column of id_column")
    # Make it None (not 'None') to use the entire file
    index_to_filter_on = 1
    value_to_filter = 'LON'

    # The K in k-means.
    #How many clusters do we assume exist? starting from 0?
    # if initial centroid are given, this is overwritten by the number of centroids in the file
    num_clusters = 8

    # When do we say the optimization has 'converged' and stop updating clusters
    # this the maximum distance any centroid has moved between 2 iterations
    opt_cutoff = 0.02

    # Generate the points from the file
    data = []
    data_names = []
    save_firstline =[]
    with open(filename_input, "rt") as f:
        #reading first line to save the column names
        save_firstline = f.readline()

        for line in f:
            x = line.split(',')
            y = [float(x[i].strip('\n') or 0) for i in list_index]
            z = ''.join(x[j]+',' for j in id_column)[:-1]
            data.append(y)
            data_names.append(z)

    # Generate the initial centroids from the file
    initial_centroid = []
    save_firstline2 = []
    initial_centroid_names = []
    initial_centroid_points = []
    initial_centroid_name_labels = 'Cluster'
    if filename_init != None:
        with open(filename_init, "rt") as f2:
            #reading first line to save the column names
            save_firstline2 = f2.readline()
            for line in f2:
                x = line.split(',')
                y = [float(x[i].strip('\n') or 0) for i in list_index]
                z = ''.join(x[j]+',' for j in id_column)[:-1]
                initial_centroid.append(y)
                initial_centroid_names.append(z)
        # creating point instances from the values read
        initial_centroid_points = [Point(initial_centroid[i],str(initial_centroid_names[i])) for i in range(num_clusters+1) ]
        # saving the labels of the ID columns
        initial_centroid_name_labels = ''.join([save_firstline2.split(',')[i]+',' for i in id_column])[:-1]
        num_clusters = len(initial_centroid_names)-1

    #adding a filter for the origin
    if value_to_filter != None:
        data = [data[i] for i in range(len(data)) if data_names[i].split(',')[index_to_filter_on] == value_to_filter ]
        data_names = [data_names[i] for i in range(len(data_names)) if data_names[i].split(',')[index_to_filter_on] == value_to_filter ]

    # Calulate size of the data
    num_points = len(data)
    dimensions = len(data[0])

    #normalizing the data => linearly [0,1] interval
    normal_data = [[0 for i in range(dimensions)] for j in range(num_points)]
    max_data = [1]*dimensions
    min_data = [0]*dimensions
    for i in range(dimensions):
        max_data[i] = max(data[j][i] for j in range(num_points))
        min_data[i] = min(data[j][i] for j in range(num_points))

    for j in range(num_points):
        for i in range(dimensions):
            if max_data[i]-min_data[i] == 0:
                normal_data[j][i] = 0
            else:
                normal_data[j][i] = (data[j][i]-min_data[i])/(max_data[i]-min_data[i])

    # Create points to cluster
    points = [Point(normal_data[i],str(data_names[i])) for i in range(num_points) ]

    # Cluster the points (that's where the magic happens)
    clusters = kmeans(points, num_clusters, opt_cutoff, initial_centroid_points)

    # Assigning names to the clusters if given otherwise numbers
    if len(initial_centroid_names) > 0:
        cluster_names = initial_centroid_names
    else:
        cluster_names = [i for i in range(num_clusters+1)]
    # Print the resulting clusters on screen
    for i,c in enumerate(clusters):
        for p in c.points:
            print('Cluster: ', str(cluster_names[i]), '\t Point :', p.name)

    # export a file with results adding cluster names as first column and all the values after
    results_file = open(filename_out, 'w')
    results_file.write(str(initial_centroid_name_labels)+','+ str(save_firstline))
    for i,c in enumerate(clusters):
        results_file.write(str(cluster_names[i])+','+centroid_name+','+str(c.centroid.coords).strip("[]")+'\n')
        for p in c.points:
            results_file.write(str(cluster_names[i])+','+str(p.name)+','+str(p.coords).strip("[]")+'\n')

class Point:
    #A point in n dimensional space, its length and name
    def __init__(self, coords, name):
        #coords - A list of values
        self.coords = coords
        self.n = len(coords)
        self.name = name

    def __repr__(self):
        return str(self.coords)

class Cluster:
    #A set of points and their centroid (which is a point)
    def __init__(self, points):
        #points - A list of point objects
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        #String representation of this object
        return str(self.points)

    def update(self, points):
        #Returns the distance between the previous centroid and the new after
        #recalculating and storing the new centroid.
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()

        shift = getDistance(old_centroid, self.centroid)
        print('moved by: ' + str(shift))
        return shift

    def calculateCentroid(self):
        if len(self.points) > 0 :
            # Finds a virtual center point for a group of n-dimensional points
            numPoints = len(self.points)
            # Get a list of all coordinates in this cluster
            coords = [p.coords for p in self.points]
            print('cluster has: '+ str(numPoints) + ' point')
            # Reformat that so all x's are together, all y'z etc.
            unzipped = zip(*coords)
            # Calculate the mean for each dimension
            centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
            return Point(centroid_coords,'Centroid')
        else:
            return self.centroid

def kmeans(points, k, cutoff, initial_centroid):

    if len(initial_centroid) == k+1:
        # use the initial centroids if given
        initial = initial_centroid
    else:
        print('Initial Centroid invalid, using random points')
        # Pick out k random points to use as our initial centroids
        initial = random.sample(points, k+1)

    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        print(' ')
        print('loop counter: '+ str(loopCounter))
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # As many times as there are clusters ...
        for i in range(clusterCount):
            print('looking at cluster number: ' +str(i))
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print("Converged after %s iterations" % loopCounter)
            break
    return clusters

def getDistance(a, b):
    #Euclidean distance between two n-dimensional points.
    #Note: This can be very slow and does not scale well
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points: one point is "+str(a.n)+" and the other one is "+str(b.n))

    ret = sum(pow((a.coords[y]-b.coords[y]), 2) for y in range(a.n))
    return math.sqrt(ret)



if __name__ == "__main__":
    main()
