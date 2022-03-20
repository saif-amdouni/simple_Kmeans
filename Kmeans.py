# imports
import numpy as np
from random import randrange
import matplotlib.pyplot as plt 

class kmeans : 
    def __init__(self,k,max_iter = 100):
        self.k = k
        self.max_iter = max_iter
    @staticmethod
    def distance(point,center):
        """calculate the euclidian distance of a point to a given center"""
        return ((point[0]-center[0])**2 + (point[1]-center[1])**2)**0.5
    @staticmethod
    def calculate_new_center(points):
        """calculate the center of a given point set"""
        return np.mean(points,axis=0)
    
    def fit(self,data):
        points =[[] for i in range(self.k)]
        # select random k points from data to be our centers
        centers = data[np.random.choice(data.shape[0], self.k, replace=False)]

        for _iter in range(self.max_iter):
            
            for data_point in data:
                distances = []
                for center in centers:
                    #calculate the distance of the datapoint to each center
                    distances.append(kmeans.distance(data_point,center))
                #asociate data point with the center of the minimal distance
                points[np.argmin(distances)].append(data_point.tolist())

            # update centers
            for center_idx in range(self.k):
                centers[center_idx] = kmeans.calculate_new_center(points[center_idx])
            # TO-DO : if centers don't change : break 
        self.centers = centers
    def predict(self,data):
        prediction = []
        for data_point in data : 
            distances = []
            for center in self.centers:
                distances.append(kmeans.distance(data_point,center))

            prediction.append(np.argmin(distances))
        return prediction

if __name__ == "__main__" :
    
    #generate data
    # 3 normal distribution
    sigma = 0.3
    interval = 5
    centers = [(randrange(interval),randrange(interval)),
                (randrange(interval),randrange(interval)),
                (randrange(interval),randrange(interval)),
                (randrange(interval),randrange(interval)),]
    n_points = 1000
    data = []
    for center in centers :
        data.append(np.column_stack((np.random.normal(center[0], sigma, n_points),np.random.normal(center[1], sigma, n_points))))
    data = np.vstack(data)

    k_means = kmeans(4,max_iter = 100)

    k_means.fit(data)
    print(f"centers : {k_means.centers}")
    plt.figure()
    plt.scatter(data[:,0],data[:,1],s=50, c=k_means.predict(data), cmap='plasma')
    plt.show()