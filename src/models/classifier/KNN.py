from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
# import BaseClassifier

# class KNN(BaseClassifier):
#     def __init__(self, n_neighbors):
#         super(KNN,self).__init__()
#         self.n_neighbors = n_neighbors

#     def predict(self, X, y):
#         nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(X)
#         distances, indices = nbrs.kneighbors(X)
#         all_top5 = np.array(y)[indices]
#         results = []
#         for i, top5 in enumerate(all_top5):
#             result = most_frequent(top5)
#             results.append(result)
        
#         return results