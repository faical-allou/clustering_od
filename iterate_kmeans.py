import os
import kmeans

if __name__ == '__main__':

    for i in range(30):

        filenames_eachrun = 'C:/Users/faica/OneDrive/Documents/dev/clustering_od/results.csv'
        results_file = 'C:/Users/faica/OneDrive/Documents/dev/clustering_od/results_agg.csv'
        kmeans.main()
        print()
        print('Run :' + str(i))

        with open(results_file, "a") as f_end:

            with open(filenames_eachrun, "rt") as f:
                #reading first line to save the column names
                firstline = f.readline()
                firstline = firstline.rstrip('\n') + ',run' + '\n'
                f_end.write(firstline)
                for line in f:
                    line = line.rstrip('\n') + ',' + str(i) + '\n'
                    f_end.write(line)

   