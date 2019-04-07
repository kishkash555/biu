import numpy as np
import load
from init_centroids import init_centroids

MAX_ITERS = 10

def main():
    X = load.load_dog()
    for k in [2,4,8,16]:
        print('k={}:'.format(k))
        k_means(X,k)


norm = lambda x: np.sqrt(np.sum(x*x,axis=1))

def arr2str(nparr):
    return ", ".join(np.array2string(np.floor(row*100)/100,formatter={'float': lambda v: "{:.2f}".format(v)}) for row in nparr)

def k_means(X,K):
    centroids = init_centroids(K)
    print("iter 0: {}".format(arr2str(centroids)))

    for i in range(MAX_ITERS):
        distances_to_centroids = np.vstack([norm(X - c) for c in centroids]) # loop over array (first pass only)
        pixel_centroids = np.argmin(distances_to_centroids,axis=0)
        centroids = [np.mean(X[pixel_centroids==k,:],axis=0) for k in range(K)]
        
        centroid_str = arr2str(centroids)
        print("iter {}: {}".format(i+1, centroid_str))

    return centroids


if __name__ == "__main__":
    main()


def scratch():
    import numpy as np; a=np.diag([1,3,4,6],1); print(a-np.array([[1,2,3,4,5]]).T)

    x=np.array([1,2,3,4],dtype=float); print(np.sqrt(x.dot(x)))

    a=np.diag([1,3,4,6],1); print(np.vstack([np.sum(a,axis=0)]))

    print([r for r in np.diag([1,3,4,6],1)])