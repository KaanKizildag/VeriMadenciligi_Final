from Algoritmalar.knn_alg import KnnAlg
from Algoritmalar.naive_bayes_alg import NaiveBayes

dt = KnnAlg()

res = dt.tahmin(x=[[4, 1, 9, 9, 2, 1, 3, 6, 1]])

print(res)
