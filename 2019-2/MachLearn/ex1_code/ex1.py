import numpy as np
import load
from init_centroids import init_centroids
import matplotlib.pyplot as plt
from collections import Counter

MAX_ITERS = 10
SAVE_EXTRAS = False

def main():
    X, img_shape = load.load_dog()
    for k in [2,4,8,16]:
        print('k={}:'.format(k))
        centroids, pixel_centroids, mean_distance = k_means(X,k)
        if SAVE_EXTRAS:
            from imageio import imwrite

            compressed_image = np.vstack(centroids)[pixel_centroids,:].reshape(img_shape)
            imwrite("compressed_{}.png".format(k),compressed_image)
            plt.clf()
            plt.plot(range(MAX_ITERS),mean_distance)
            plt.title('Loss, K={}'.format(k))
            plt.xlabel('iteration')
            plt.ylabel('mean distance')
            plt.savefig("loss_curve_{}.png".format(k))



norm = lambda x: np.sum(x*x,axis=1)

#def arr2str(nparr):
#    return ", ".join(np.array2string(np.floor(row*100)/100, separator=', ') for row in nparr)

def arr2str2(nparr):
    entry_formatter = lambda x: str(x) if str(x)!='0.0' else '0.'
    row_formatter = lambda x: "["+", ".join(entry_formatter(x[i]) for i in range(len(x)))+"]"
    return ", ".join(row_formatter(np.floor(row*100)/100) for row in nparr)

def k_means(X,K):
    centroids = init_centroids(K)
    print("iter 0: {}".format(arr2str2(centroids)))
    mean_distance=[]
    for i in range(MAX_ITERS):
        distances_to_centroids = np.vstack([norm(X - c) for c in centroids]) # loop over array (first pass only)
        pixel_centroids = np.argmin(distances_to_centroids,axis=0)

        centroids = [np.mean(X[pixel_centroids==k,:],axis=0) for k in range(K)]
        
        total_dist=0.
        for col,row in enumerate(pixel_centroids):
            total_dist+=distances_to_centroids[row,col]
        mean_distance.append(total_dist/pixel_centroids.shape[0])
        centroid_str = arr2str2(centroids)
        print("iter {}: {}".format(i+1, centroid_str))

    return centroids, pixel_centroids, mean_distance


if __name__ == "__main__":
    main()


def scratch():
    import numpy as np; a=np.diag([1,3,4,6],1); print(a-np.array([[1,2,3,4,5]]).T)

    x=np.array([1,2,3,4],dtype=float); print(np.sqrt(x.dot(x)))

    a=np.diag([1,3,4,6],1); print(np.vstack([np.sum(a,axis=0)]))

    print([r for r in np.diag([1,3,4,6],1)])

    formatter={'float': lambda v: "{:.2f}".format(v)},