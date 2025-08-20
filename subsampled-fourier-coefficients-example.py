# Author: Jonathan Siegel
#
# Demonstrates the recovery of a continuous, periodic, two-dimensional function of bounded variation from appropriately subsampled Fourier coefficients.

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import math
import itertools

def generate_ground_truth_image(k):
  """Generates the coordinates and (exact) Fourier coefficients of the ground truth image.
     The ground truth is given by f(x,y) = - 0.75 * 1(|x-3| <= 2 and |y-3| <= 1) - 1.0 * 1(|x-3.0| + |y-4.0| <= 1)
     Here 1() denotes the indicator function of a set and f is defined periodically on [0,2pi] x [0,2pi]. 

     Args:
       k: The sampling dimension of the image is (2^k+1) by (2^k+1).

     Returns:
       Matrix of ground truth values and matrix of ground truth Fourier coefficients.
  """
  n = 2**k + 1
  image = np.zeros((n,n))
  fourier = np.zeros((n,n), dtype=complex)
  grid = np.linspace(0, 2.0*np.pi, n)
  xgrid, ygrid = np.meshgrid(grid, grid)
  freq = fft.fftfreq(n, d=1.0/n)
  xfreq, yfreq = np.meshgrid(freq, freq)

  # Add the first indicator function.
  image -= 0.75 * ((xgrid >= 1) & (xgrid <= 5) & (ygrid >= 2) & (ygrid <= 4))
  fourier -= 0.75 * np.where(xfreq != 0, (-1.0 / (1j * xfreq)) * (np.exp(-1j * xfreq * 5.0) - np.exp(-1j * xfreq * 1.0)), 5.0 - 1.0) \
                  * np.where(yfreq != 0, (-1.0 / (1j * yfreq)) * (np.exp(-1j * yfreq * 4.0) - np.exp(-1j * yfreq * 2.0)), 4.0 - 2.0)

  # Add the second indicator function.
  image -= 1.0 * (np.abs(xgrid - 3.0) + np.abs(ygrid - 4.0) <= 1)
  fourier -= 1.0 * 2 * np.where(xfreq != yfreq, (-1.0 / (1j * (xfreq - yfreq))) * (np.exp(-1j * (xfreq - yfreq) * 0.0) - np.exp(-1j * (xfreq - yfreq) * (-1.0))), 0.0 - (-1.0)) \
                  * np.where(xfreq + yfreq != 0, (-1.0 / (1j * (xfreq + yfreq))) * (np.exp(-1j * (xfreq + yfreq) * 4.0) - np.exp(-1j * (xfreq + yfreq) * 3.0)), 4.0 - 3.0)

  return image, fourier

def linear_reconstruction(fourier_coefficients, k, smoothing=True):
  """Reconstructs an image by truncating the Fourier coefficients at level k.
     Uses de la Vallee Poisson smoothing.

     Args:
       k: The level of truncation is 2^k + 1

     Returns:
       Reconstructed image.
  """
  n1 = 2**k + 1
  fourier_small = np.zeros((n1,n1), dtype='complex')
  indices = [i - (n1 - 1)//2 for i in range(n1)]
  x_indices, y_indices = np.meshgrid(indices, indices)
  fourier_small[x_indices, y_indices] = fourier_coefficients[x_indices, y_indices]
  de_la_vallee_poisson_factor = np.minimum(1, (2**(k-1) - np.abs(x_indices)) / (2**(k-2))) * \
                                np.minimum(1, (2**(k-1) - np.abs(y_indices)) / (2**(k-2)))
  if smoothing:
    fourier_small = fourier_small * de_la_vallee_poisson_factor[x_indices, y_indices]
  k_large = int(math.floor(math.log2(fourier_coefficients.shape[0])))
  return inverse_fourier_transform(fourier_small, k_large)

def generate_square_mask(k, k_large):
  """Generates a mask which selects the lowest (2^k + 1) by (2^k + 1) Fourier modes out of
     (2^k_large + 1) by (2^k_large + 1) total modes.

     Args:
       k: Total number of samples is (2^k + 1) by (2^k + 1)
       k_large: Total number of Fourier modes is (2^k_large + 1) by (2^k_large + 1)

     Returns:
       2d mask sampling the lowest modes.
  """
  n = 2**k_large + 1
  mask = np.zeros((n, n))
  n_small = 2**k + 1
  ind = [i - (n_small - 1)//2 for i in range(n_small)]
  x_ind, y_ind = np.meshgrid(ind, ind)
  mask[x_ind, y_ind] = 1
  return mask

def generate_uniform_random_mask(k, num_modes):
  """Generates a uniformly random set of Fourier modes at which to sample.

  Args:
    k: The size of the Fourier grid is (2^k + 1) by (2^k + 1)
    num_modes: The number of modes to randomly select.

  Returns:
    2d mask indicating the random Fourier sample points.
  """
  n = 2**k + 1
  mask = np.zeros((n, n))
  rng = np.random.default_rng()
  indices = np.array([[i - (n - 1)/2, j - (n - 1)/2] for (i, j) in itertools.product(range(n), range(n)) \
                       if (i > (n - 1)/2 or (i == (n - 1)/2 and j > (n - 1)/2))], dtype='int')
  rand_indices = np.transpose(rng.choice(indices, num_modes, replace=False))
  mask[rand_indices[0], rand_indices[1]] = 1
  mask[-1 * rand_indices[0], -1 * rand_indices[1]] = 1
  return mask 

def generate_random_tapered_mask(k, taper_factor, k_large):
  """Generates a random set of Fourier coefficients at which to sample. The entire block up to
     (2^k + 1) by (2^k + 1) is taken. Subsequent higher frequency blocks are subsampled in such a
     way that the number of samples taken decreases by a factor of taper_factor in each step. Once this
     factor drops below 1, or the maximum size k_large has been reached, the sampling is completed.

     Args:
       k: The last block which is completely taken is (2^k + 1) by (2^k + 1)
       taper_factor: geometric rate of decrease of number of samples
       k_large: The k corresponding to the output size of the mask

     Returns:
       2d mask indicating where the transform is sampled.
  """
  n = 2**k + 1
  mask = np.ones((n, n))
  count = n**2
  count = int(math.floor(count * taper_factor))
  k = k+1
  rng = np.random.default_rng()
  prev_indices = []
  while (count > 1 and k <= k_large):
    n_new = 2**k + 1
    mask_new = np.zeros((n_new, n_new))

    # Fill in the mask from the previous round.
    ind = [i - (n - 1)//2 for i in range(n)]
    x_ind, y_ind = np.meshgrid(ind, ind)
    mask_new[x_ind, y_ind] = mask[x_ind, y_ind]

    indices = [[i - (n_new - 1)/2, j - (n_new - 1)/2] for (i, j) in itertools.product(range(n_new), range(n_new)) \
                       if (abs(i - (n_new - 1)/2) > (n-1)/2 or abs(j - (n_new - 1)/2) > (n-1)/2) and (i > (n_new - 1)/2 or (i == (n_new - 1)/2 and j > (n_new - 1)/2))]
    # Select random frequencies from two consecutive dyadic blocks (to obtain a random selection from overlapping blocks).
    rand_indices = np.transpose(rng.choice(np.array(indices + prev_indices, dtype='int'), int(math.floor(count / 2)), replace=False))
    mask_new[rand_indices[0], rand_indices[1]] = 1
    mask_new[-1 * rand_indices[0], -1 * rand_indices[1]] = 1
    count = int(math.floor(count * taper_factor))
    n = n_new
    mask = mask_new
    # Add unselected indices back to be potentially selected in next overlapping block.
    prev_indices = [ind for ind in indices if mask[int(ind[0]), int(ind[1])] == 0]
    k = k+1
  if k <= k_large:
    n_new = 2**k_large + 1
    mask_new = np.zeros((n_new, n_new))

    # Fill in the mask from the previous round.
    ind = [i - (n - 1)//2 for i in range(n)]
    x_ind, y_ind = np.meshgrid(ind, ind)
    mask_new[x_ind, y_ind] = mask[x_ind, y_ind]

    mask = mask_new
  return mask

def inverse_fourier_transform(fourier, k):
  """Calculates the inverse Fourier transform given continuous Fourier coefficients of f.

  Args:
    fourier: The continuous Fourier coefficients of f.
    k: The function is reconstructed on a grid of size 2^k + 1.

  Returns:
    values of the 2d function at a grid with size 2^k+1.
  """
  n1 = fourier.shape[0]
  n2 = 2**k + 1
  enlarged_fourier = np.zeros((n2, n2), dtype='complex')
  indices = [i - (n1 - 1)//2 for i in range(n1)]
  x_indices, y_indices = np.meshgrid(indices, indices)
  enlarged_fourier[x_indices, y_indices] = fourier[x_indices, y_indices]
  return fft.ifft2(enlarged_fourier, norm='forward') / (4.0 * np.pi**2)

def fourier_transform(f, k):
  """Calculates the Fourier transform a trigonometric polynomial. This is the transpose of the inverse_fourier_transform function.
  
  Args:
    f: The input function f.
    k: The Fourier coefficients are evaluated on a grid of size 2^k + 1.

  Returns:
    Fourier coefficients of f up to 2^k + 1.
  """
  n1 = f.shape[0]
  n2 = 2**k + 1
  fft_coeffs = fft.fft2(f, norm='backward') / (4.0 * np.pi**2)
  fourier = np.zeros((n2, n2), dtype='complex')
  indices = [i - (n2 - 1)//2 for i in range(n2)]
  x_indices, y_indices = np.meshgrid(indices, indices)
  fourier[x_indices, y_indices] = fft_coeffs[x_indices, y_indices]
  return fourier
 

def minimal_BV_norm_reconstruction(fourier_coefficients, mask, step_size = 0.2):
  """Reconstructs a function by minimizing the BV norm subject to the given Fourier measurements.
     The problem is discretized on a doubly oversampled grid and the resulting discrete optimization
     problem is solved using ADMM.

  Args:
    fourier_coefficients: Input Fourier coefficients
    mask: Mask giving the positions of measured Fourier coefficients (the rest will be inferred via optimization).
    step_size: step size (default: 0.2)

  Returns:
    Reconstructed image on grid of the same size as the input Fourier coefficients.
  """
  # Set the Fourier coefficients outside of the mask equal to 0.
  fourier_coefficients = fourier_coefficients * mask

  n1 = fourier_coefficients.shape[0]
  k = int(math.floor(math.log2(n1)))
  indices = 1.0j * fft.fftfreq(n1, d=1.0/n1)
  x_factor, y_factor = np.meshgrid(indices, indices)

  # Construct derivative variables on a larger grid (for estimating the l1-norm)
  n2 = 2 * n1 - 1
  lambda_x = np.zeros((n2, n2), dtype = 'complex')
  lambda_y = np.zeros((n2, n2), dtype = 'complex')

  # Run ADMM for 2000 steps. Probably overkill, but want to ensure convergence.
  print('Running ADMM for iterations:')
  for i in range(2000):
    print(i)
    x_deriv = inverse_fourier_transform(fourier_coefficients * x_factor, k+1) - lambda_x
    y_deriv = inverse_fourier_transform(fourier_coefficients * y_factor, k+1) - lambda_y
    norm_deriv = np.sqrt(x_deriv * np.conjugate(x_deriv) + y_deriv * np.conjugate(y_deriv))
    norm_deriv = np.real(norm_deriv)
    shrink_factor = np.where(norm_deriv > 0, np.maximum(norm_deriv - step_size, 0) / norm_deriv, 0.0)
    x_deriv = shrink_factor * x_deriv
    y_deriv = shrink_factor * y_deriv
    new_values = fourier_transform(x_deriv + lambda_x, k) * np.conjugate(x_factor) \
               + fourier_transform(y_deriv + lambda_y, k) * np.conjugate(y_factor)
    div_factor = (x_factor * np.conjugate(x_factor) + y_factor * np.conjugate(y_factor)) * (n2**2) / (16.0 * np.pi**4)
    div_factor[0,0] = 1.0
    new_values = new_values / div_factor
    fourier_coefficients = fourier_coefficients * mask + new_values * (1 - mask)
    lambda_x = lambda_x + x_deriv - inverse_fourier_transform(fourier_coefficients * x_factor, k+1)
    lambda_y = lambda_y + y_deriv - inverse_fourier_transform(fourier_coefficients * y_factor, k+1)
  return inverse_fourier_transform(fourier_coefficients, k)

def plot_image(img, title=""):
    plt.imshow(img, cmap='Greys_r')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title(title)
    plt.show()

def main():
  image, fourier = generate_ground_truth_image(10)
  plot_image(image, "Ground Truth")

  # Reconstruct using lowest Fourier modes.
  k = 4
  plot_image(np.real(linear_reconstruction(fourier, k, smoothing=False)), "Reconstruction by Summing Fourier Series")
  plot_image(np.real(linear_reconstruction(fourier, k, smoothing=True)), "Reconstruction by de la Vallee Poisson Summation")
  mask = generate_square_mask(k, 10)
  plot_image(np.real(minimal_BV_norm_reconstruction(fourier, mask)), "Reconstruction by BV-norm minimization (lowest frequencies)")

  # Reconstruct using random Fourier modes.
  mask = generate_uniform_random_mask(10, (2**k+1)*(2**k+1))
  plot_image(np.real(minimal_BV_norm_reconstruction(fourier, mask)), "Reconstruction by BV-norm minimization (uniform random sub-sampling)")

  # Reconstruct using subsampled Fourier modes of the same size.
  mask = generate_random_tapered_mask(3, 0.765, 10)
  # Print the number of frequencies sampled.
  print(np.sum(mask))
  plot_image(np.real(minimal_BV_norm_reconstruction(fourier, mask)), "Reconstruction by BV-norm minimization (hierarchical random sub-sampling)")
  

if __name__ == '__main__':
  main()
