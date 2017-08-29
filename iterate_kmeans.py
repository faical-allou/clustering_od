import os
import kmeans

if __name__ == '__main__':

    for i in range(20):
        filename = 'C:/Users/faicalallou/Documents/Dev/clustering_runs'
        kmeans.main()
        os.rename(filename+'.csv', filename+str(i)+'.csv')
