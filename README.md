## Comparing-FFT-vs.-DCT
Fourier transform vs. discrete cosine transform

For data compression, Principle Component Decomposition (PCA) and Singular Value Decomposition (SVD) is often used. 

In induatrial grade image and audio compression (jpeg and mp3), Discrete Cosine Transform (DCT) is a more efficient way. DCT is similar to FFT or DFT in that it identified the key frequencies. DCT differs from FFT/DFT in that it requires data to be symmetrical around origin. 
<img width="823" height="184" alt="image" src="https://github.com/user-attachments/assets/71d29ebf-e3fd-42cb-b74a-61ff0f27ec36" style="width:60%;"/>


<img width="1041" height="243" alt="image" src="https://github.com/user-attachments/assets/a6ce9fb5-1e16-409d-b0b9-ad7f4d2d7fca" style="width:60%;"/>


```python

def idct_iii_ortho(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    k = np.arange(N)
    n = np.arange(N)
    C = np.cos(np.pi * (n[:, None] + 0.5) * k[None, :] / N)
    alpha = np.ones(N); alpha[0] = 1/np.sqrt(2)
    x_rec = np.sqrt(2/N) * (C @ (alpha * X))
    return x_rec
```

Note: numpy array display format is different from the computation format. For example, a 1D array is displayed horizontally. However, in calculation, it's treated as a column vector. Just like in this line

```python
 x_rec = np.sqrt(2/N) * (C @ (alpha * X))
```
where 
```python
x_rec
X
```
are both column vector of size N
```python
C
```
is NxN matrix.
If taking arrays as they are displayed to do matrix multiplication, it would not make sense. 

## Covariance matrix

Covariance matrix is 
$$
\sigma = \frac{1}{n-1}\sum_{i=1}^n(x_i - \bar{x})^2
$$


## numpy tips
A @ B is matrix multiplication, it's equivalent to np.dot(A, B)


$$
C = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})(X_i - \bar{X})^T
$$

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots \\
a_{21} & a_{22} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$
