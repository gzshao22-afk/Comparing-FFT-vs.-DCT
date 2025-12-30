
import numpy as np
import matplotlib.pyplot as plt

# Generate a real 1D signal (sum of cosines + noise)
np.random.seed(0)
N = 256
n = np.arange(N)
sig = (1.8*np.cos(2*np.pi*4*n/N) +
       0.9*np.cos(2*np.pi*17*n/N + 0.4) +
       0.4*np.cos(2*np.pi*35*n/N + 0.9))
sig += 0.12*np.random.randn(N)

# FFT (complex)
X_fft = np.fft.fft(sig)
half = N//2
freqs = np.fft.fftfreq(N, d=1.0)[:half]
mag_fft = np.abs(X_fft[:half])


# DCT-II (orthonormal) implemented explicitly
def dct_ii_ortho(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    k = np.arange(N)
    n = np.arange(N)
    C = np.cos(np.pi * (n[:, None] + 0.5) * k[None, :] / N)
    alpha = np.ones(N); alpha[0] = 1/np.sqrt(2)  # DC scaling for orthonormality
    X = np.sqrt(2/N) * (C.T @ x)
    X *= alpha
    return X

def idct_iii_ortho(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    k = np.arange(N)
    n = np.arange(N)
    C = np.cos(np.pi * (n[:, None] + 0.5) * k[None, :] / N)
    alpha = np.ones(N); alpha[0] = 1/np.sqrt(2)
    x_rec = np.sqrt(2/N) * (C @ (alpha * X))
    return x_rec

X_dct = dct_ii_ortho(sig)

# Energy compaction: fraction of energy in top-k coefficients
def energy_fraction(coeffs, k):
    idx = np.argsort(np.abs(coeffs))[::-1]
    top_energy = np.sum(np.abs(coeffs[idx[:k]])**2)
    total_energy = np.sum(np.abs(coeffs)**2)
    return top_energy/total_energy

ks = [4, 8, 16, 32, 64]
energy_fft = [energy_fraction(X_fft[:half], k) for k in ks]  # half-spectrum
energy_dct = [energy_fraction(X_dct, k) for k in ks]

# Reconstruction with top-k
def reconstruct_dct_top_k(Xdct, k):
    idx = np.argsort(np.abs(Xdct))[::-1]
    mask = np.zeros_like(Xdct)
    mask[idx[:k]] = Xdct[idx[:k]]
    return idct_iii_ortho(mask)

def reconstruct_fft_top_k(Xfft, k):
    # keep k largest magnitudes across full spectrum, enforce conjugate symmetry
    N = len(Xfft)
    mags = np.abs(Xfft)
    idx = np.argsort(mags)[::-1]
    keep = idx[:k]
    mask = np.zeros_like(Xfft)
    mask[keep] = Xfft[keep]
    for i in keep:
        if i == 0 or (N%2==0 and i==N//2):  # DC and Nyquist
            continue
        j = (-i) % N
        mask[j] = np.conj(mask[i])
    return np.fft.ifft(mask).real

recon_dct = {k: reconstruct_dct_top_k(X_dct, k) for k in ks}
recon_fft = {k: reconstruct_fft_top_k(X_fft, k) for k in ks}

err_dct = {k: np.linalg.norm(sig - recon_dct[k]) / np.linalg.norm(sig) for k in ks}
err_fft = {k: np.linalg.norm(sig - recon_fft[k]) / np.linalg.norm(sig) for k in ks}

# Full reconstruction checks
sig_rec_dct_full = idct_iii_ortho(X_dct)
err_full_dct = np.linalg.norm(sig - sig_rec_dct_full) / np.linalg.norm(sig)

sig_rec_fft_full = np.fft.ifft(X_fft).real
err_full_fft = np.linalg.norm(sig - sig_rec_fft_full) / np.linalg.norm(sig)

# Plots
plt.figure(figsize=(13,9))
plt.subplot(2,2,1)
plt.plot(sig, label='Signal')
plt.title('Original 1D signal')
plt.legend()

plt.subplot(2,2,2)
plt.plot(np.arange(N), X_dct, label='DCT-II coeffs')
plt.title('DCT coefficients (orthonormal)')
plt.xlabel('k'); plt.legend()

plt.subplot(2,2,3)
plt.plot(freqs, mag_fft, label='|FFT| (half-spectrum)')
plt.title('FFT magnitude (non-negative frequencies)')
plt.xlabel('frequency index'); plt.legend()

plt.subplot(2,2,4)
width = 0.35; xpos = np.arange(len(ks))
plt.bar(xpos - width/2, energy_dct, width, label='DCT top-k energy')
plt.bar(xpos + width/2, energy_fft, width, label='FFT top-k energy (half)')
plt.xticks(xpos, ks); plt.ylim(0,1)
plt.title('Energy compaction comparison')
plt.xlabel('k kept'); plt.ylabel('fraction of energy')
plt.legend()
plt.tight_layout()
plt.show()

print('Full reconstruction relative error:')
print('DCT:', err_full_dct, ' | FFT:', err_full_fft)
print('Top-k relative errors (smaller is better):')
print('DCT:', err_dct)
print('FFT:', err_fft)
