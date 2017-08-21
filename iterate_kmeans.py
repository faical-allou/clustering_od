import os
import kmeans

if __name__ == '__main__':

    for i in range(10):
        filename = 'C:/Users/faicalallou/Documents/Dev/clustering_runs2'
        kmeans.main()
        os.rename(filename+'.csv', filename+str(i)+'.csv')
