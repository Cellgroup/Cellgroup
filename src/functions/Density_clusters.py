import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import random

from GradientClustering import gradientClustering, plotClusteringReachability, filterLargeClusters, \
    mergeSimilarClusters


def Density_clusters(input_data, w, max_points_ratio, cluster_similarity_threshold, min_points, epsilon):

    # Plot input data
    plotPoints(input_data, title='Input data')

    t1 = time.process_time()

    # Run OPTICS ordering
    ordered_list = runOPTICS(input_data, epsilon, min_points)

    print('Total time for processing', time.process_time() - t1, 's')

    print('Ordered list')
    print('Point index [Processed, reachability dist, code dist, input data ... ]')

    print(ordered_list[:, 1])
    # Check the shape of the input data
    print('Shape of input data:', input_data.shape)

    # Check the values of the OPTICS parameters
    print('OPTICS parameters: min_points =', min_points, ', epsilon =', epsilon)

    # Check the shape and contents of the ordered_list
    print('Shape of ordered_list:', ordered_list.shape)
    # print('Contents of ordered_list:', ordered_list)

    # Check the shape and contents of the reachability array
    reachability = ordered_list[:, 1]
    print('Shape of reachability array:', reachability.shape)
    # print('Contents of reachability array:', reachability)

    # Plot the reachability diagram
    plotClusteringReachability(ordered_list[:, 1])

    # Do the gradient clustering
    clusters = gradientClustering(ordered_list[:, 1], min_points, t, w)

    # Remove very large clusters
    filtered_clusters = filterLargeClusters(clusters, len(ordered_list), max_points_ratio)

    print('TOTAL BEFORE MERGING', len(filtered_clusters))

    # Plot the results, reachability diagram
    plotClusteringReachability(ordered_list[:, 1], filtered_clusters)

    # Merge similar clusters by looking at the ratio of their intersection and their total number
    filtered_clusters = mergeSimilarClusters(filtered_clusters, cluster_similarity_threshold)

    print('TOTAL POINTS', len(ordered_list[:, 1]))
    print('CLUSTERS')
    for cluster in filtered_clusters:
        members = ordered_list[cluster][:, 3:]

        x_mean = np.mean(members[:, 0])
        x_std = np.std(members[:, 0])

        y_mean = np.mean(members[:, 1])
        y_std = np.std(members[:, 1])

        print('------------------------------------------')

        print('Size, X mean +/- stddev, Y mean +/- stddev')
        print(len(cluster), x_mean, x_std, y_mean, y_std)
        print('Members:')
        print(members)

    # Plot the results, reachability diagram
    plotClusteringReachability(ordered_list[:, 1], filtered_clusters)

    # Plot the final results
    plotPoints(input_data, filtered_clusters, title='Final results')