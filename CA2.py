# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create panda of each category
animals = pd.read_csv("animals", sep = " ", header = None).to_numpy()
countries = pd.read_csv("countries", sep = " ", header = None).to_numpy()
fruits = pd.read_csv("fruits", sep = " ", header = None).to_numpy()
veggies = pd.read_csv("veggies", sep = " ", header = None).to_numpy()

# Change first column to label. 0 = animals, 1 = countries, 2 = fruits, 3 = veggies
animals[:, 0] = 0
countries[:, 0] = 1
fruits[:, 0] = 2
veggies[:, 0] = 3

# Join all data
data = np.concatenate([animals, countries, fruits, veggies])

# Store all labels in array
labels = np.copy(data[:, 0])

# Dataset to use is array of the data, without the labels
dataset = np.delete(data, 0, 1).astype(float)

def initialCentroids(dataset, k):
    '''
    Finds the initial centroids in the dataset to cluster objects to.
    
    Parameters:
    dataset - Numpy array.
    k - Integer. Number of clusters.
    
    Return:
    centroids as numpy array.
    '''
    #Pick random centroids from the dataset. Set random seed
    np.random.seed(10)

    #Total number of objects to be assigned
    numOfObjects = len(dataset)

    #Get the indices of the objects that we pick as initial centroids
    centroidsInd = np.random.choice(numOfObjects, k, replace=False)

    #Make list of initial centroids
    centroids = []
    for index in centroidsInd:
        centroids.append(dataset[index])

    return centroids

def clustering(dataset, k, method, norm):
    '''
    Clusters objects in a dataset into k clusters using.
    either K-Means or K-Medians Returns a tuple
    containing the b-cubed precision, recall, fScore
    and k.

    Parameters:
    dataset - Numpy array.
    k - Integer. Number of clusters.
    method - String. Either kMeans or kMedians
    norm - Use True to normalise the objects to l2 length.
    False to cluster unnormalised data.

    Return:
    precision, recall fScore, k
    '''
    #Normalize data
    if norm == True:
        dataset = dataset / np.linalg.norm(dataset)

    #Pick initial centroids
    centroids = initialCentroids(dataset, k)

    # Create copies of the centroids for updating
    oldCentroids = np.zeros(np.shape(centroids))
    newCentroids = np.copy(centroids)

    # Create a blank distance and cluster assignment object to hold results
    clusters = np.zeros(dataset.shape[0])
    # Create error object
    error = np.linalg.norm(newCentroids - oldCentroids)

    # Whilst there is an error:
    while error != 0:
        # dist is array of zeros with same number of rows as dataset, and k columns. So if k = 4, dist will be 300 rows and 4 columns. Use this to store distances
        # from each point to each centroid
        dist = np.zeros([dataset.shape[0], k])

        # Calculate the Euclidean distance from each point to each centroid.
        for i in range(len(centroids)):
            #For kMeans
            if method == "kMeans":
                #All rows in column i = euclidean distance between
                # Distance from object to first centroid is in first column, to second in second column etc.
                # So we can see all distances from all objects to all centroids
                dist[:, i] = np.linalg.norm(dataset - newCentroids[i], axis=1)
            elif method == "kMedians":
                # Calculate the Manhattan distance from each point to each centroid.

                # All rows in column i = Manhattan distance between
                # Distance from object to first centroid is in first column, to second in second column etc.
                # So we can see all distances from all objects to all centroids

                for i in range(len(centroids)):
                    dist[:, i] = np.sum(np.abs(dataset - newCentroids[i]), axis=1)
        
        # Calculate the cluster assignment. Find the minimum value (distance) in each row.
        # Then input index of that value. This is the cluster that the object belongs to.
        clusters = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids to the old centroids object
        oldCentroids = np.copy(newCentroids)

        # For kMeans
        if method == "kMeans":
            # Calculate the mean to re-adjust the cluster centroids
            # newCentroid[m] will be the mean of points in dataset that correspond to clusters that == m.
            # So for cluster 1, we find the objects in clusters whose minimum distance was found to be from the first
            # centroid, calculate mean. And so on.
            for j in range(k):
                newCentroids[j] = np.mean(dataset[clusters == j], axis = 0)
        elif method == "kMedians":
            # Calculate the median to re-adjust the cluster centroids
            # newCentroid[m] will be the median of points in dataset that correspond to clusters that == m.
            # So for cluster 1, we find the objects in clusters whose minimum distance was found to be from the first
            # centroid, calculate mean. And so on.
            for j in range(k):
                newCentroids[j] = np.median(dataset[clusters == j], axis = 0)

        # Re-calculate the error
        error = np.linalg.norm(np.array(newCentroids) - np.array(oldCentroids))
    
    #Assign the final clusters and centroids to their pairs
    finalPairs = list(list(x) for x in zip(labels, clusters))

    # Make cluster dictionary
    clusterDict = {}
    # Add keys to cluster dictionary. Keys are just the number of clusters, so use k
    for l in range(k):
        clusterDict[l] = []

    # For each key in the cluster dictionary, if the second number in the pairs == key. Add to cluster dictionary
    for key in clusterDict:
        for tup in finalPairs:
            if tup[1] == key:
                clusterDict[key].append(tup)

    #Remove second element of each list in cluster dictionary values
    for m in range(k):
        for obj in clusterDict[m]:
            obj.remove(obj[1])
    
    # print("clusterDict after removing second elements:", clusterDict)
    # Find b-cubed precision
    precision = 0
    # Go through each key in dictionary. Find number of objects with each label 0-3. Divide by the length of the key. Multiply by number of objects in that cluster
    # with label
    for key in clusterDict.keys():
        for n in range(4):
            precision += (clusterDict[key].count([n])/len(clusterDict[key]))*clusterDict[key].count([n])

    # Divide by number of objects
    precision = precision/len(dataset)
    
    #Recall
    # Set initial recall, denominator and denominator list
    recall = 0
    denom = 0
    denomList = []
    # Go through each key in dictionary find how many of each true class labels there are. Add them to list of denominators
    for o in range(4):
        for p in clusterDict.values():
            denom += sum (1 for v in p if v == [o])
        denomList.append(denom)
        denom = 0

    # For each key in dictionary, find number of objects with each label 0-3. Divide by the total number of that object in the dataset.
    # Multiply by number of objects with that label in the cluster
    for q in clusterDict.keys():
        for r in range(4):
            recall += (clusterDict[q].count([r])/denomList[r])*clusterDict[q].count([r])

    # Divide by total number of objects in dataset
    recall = recall/len(dataset)

    # F-Score
    fScore = (2*precision*recall)/(precision + recall)

    #Return the precision, recall, fScore and k
    return precision, recall, fScore, k

def plotScores(dataset, k, method, norm):
    '''
    Clusters using either K-Means or K-Medians.
    Plots values of k in x-axis and values of
    b-cubed precision, recall and fScore in y-axis.
    
    Parameters:
    dataset - Numpy array.
    k - Integer. Maximum value of k to plot.
    method - String. Either kMeans or kMedians.
    norm - Use True to normalise the objects to l2 length.
    False to cluster unnormalised data.
    
    Return:
    Line chart plotting b-cubed precision, recall and fScore
    for clustering for values 1-k.
    List of fScores and corresponding values of k
    '''
    #List of scores with k values
    listOfPlots = []

    # Calculate scores for all values of k
    for i in range(k):
        listOfPlots.append(clustering(dataset, i+1, method, norm))
    
    #Make lists to store x and y values
    precisionPlot = []
    recallPlot = []
    fScorePlot = []

    #Add x and y values to lists
    for i in listOfPlots:
        #Add precision score
        pair = (i[3], i[0])
        precisionPlot.append(pair)
        #Add recall score
        pair = (i[3], i[1])
        recallPlot.append(pair)
        #Add fScore
        pair = (i[3], i[2])
        fScorePlot.append(pair)

    print("fScores:", fScorePlot)

    # line 1 points
    x1 = [x[0] for x in precisionPlot]
    y1 = [x[1] for x in precisionPlot]
    # plotting the line 1 points 
    plt.plot(x1, y1, label = "Precision")
    # line 2 points
    x2 = [x[0] for x in recallPlot]
    y2 = [x[1] for x in recallPlot]
    # plotting the line 2 points 
    plt.plot(x2, y2, label = "Recall")
    # line 3 points
    x3 = [x[0] for x in fScorePlot]
    y3 = [x[1] for x in fScorePlot]
    # plotting the line 3 points
    plt.plot(x3, y3, label = "FScore")
    # Set the axis label of the current axis.
    plt.xlabel('K')
    plt.ylabel('Score')
    # Set a title of the current axes.
    if method == "kMeans" and norm == True:
        plt.title('K-Means Clustering B-Cubed Scores for normalised data')
    elif method == "kMeans" and norm == False:
        plt.title('K-Means Clustering B-Cubed Scores for unnormalised data')
    elif method == "kMedians" and norm == True:
        plt.title('K-Medians Clustering B-Cubed Scores for normalised data')
    elif method == "kMedians" and norm == False:
        plt.title('K-Medians Clustering B-Cubed Scores for unnormalised data')
    # Show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

userInput = 2
while userInput != 0:
    userInput = int(input("Press 1 for K-Means. Press 2 for K-Medians. Press 0 to quit."))

    if userInput == 1:
        userInput = int(input("Press 1 to use unnormalised data. Press 2 to use L2 normalised data. Press 0 to quit."))
        if userInput == 1:
            plotScores(dataset, 9, "kMeans", False)
        elif userInput == 2:
            plotScores(dataset, 9, "kMeans", True)
        elif userInput == 0:
            print("You have quit the program")
            exit()
        else:
            print("You must enter either 1, 2 or 0.")

    elif userInput == 2:
        userInput = int(input("Press 1 to use unnormalised data. Press 2 to use L2 normalised data. Press 0 to quit."))
        if userInput == 1:
            plotScores(dataset, 9, "kMedians", False)
        elif userInput == 2:
            plotScores(dataset, 9, "kMedians", True)
        elif userInput == 0:
            print("You have quit the program")
            exit()
        else:
            print("You must enter either 1, 2 or 0.")
    
    elif userInput == 0:
        print("You have quit the program")
        exit()

    else:
        print("You must enter either 1, 2 or 0.")