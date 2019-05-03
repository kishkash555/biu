## Exercise 1 Part 2
MY implementation of K-Means in numpy is as follows:
1. calculate `Y=X-Zk` where `X` is an N-by-3 array (rows are pixels, columns are for the 3 color channels). `Zk` is one of the centroids.
1. Take the norm of each row, i.e. $D_i=(y_{i1})^2 + (y_{i2})^2 + (y_{i3})^2$.

1. Repeat for the rest of the centroids. Stack vertically, creating a K-by-N array (each column represents the distances of a pixel from all centroids)
1. Take the `argmin` of each column, to find which centroid the pixel is now closest to, and therfore "belongs" to.
1. Update the mean distance between each centroid and the pixels that are assigned to it. 
1. Repeat step 1-5 10 times. For K=2, the process converged after just 6 iterations.

The `k_means` function outputs the centroid locations, the index of the centroid assigned to each pixel, and the mean distance in each iteration.

There was no need to run multiple times since the centroid initialization is fixed and the run is  determinstic.

<div style="page-break-after: always;"></div>

### Mean Distance Plot and Image After Compression for k=2
![plot2](loss_curve_2.png)

![dog2](compressed_2.png)


<div style="page-break-after: always;"></div>

### Mean Distance Plot and Image After Compression for k=4
![plot4](loss_curve_4.png)

![dog4](compressed_4.png)

<div style="page-break-after: always;"></div>

### Mean Distance Plot and Image After Compression for k=8
![plot8](loss_curve_8.png)

![dog8](compressed_8.png)

<div style="page-break-after: always;"></div>

### Mean Distance Plot and Image After Compression for k=16
![plot16](loss_curve_16.png)

![dog16](compressed_16.png)