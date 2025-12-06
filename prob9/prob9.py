import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, weibull_min, geom

# I used ChatGPT to help me generate the noise data, as I didn't have 
# time to ask my mentor for permission to use the data we captured in our experiments.
# the student wouldn't see that it got fitted to a Weibull...
########################################################
# CHATGPT CODE TO GENERATE THE NOISE DATA
np.random.seed(42)
n = 5000

# Define discrete bin edges
k = 90
bins = np.linspace(0, 5, k + 1)   # noise lives on [0, 5]
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Define a Weibull SHAPE for the discrete pmf (not continuous)
shape_k = 1.5
scale = 1.0

# Create *discrete* Weibull pmf over bin centers
weib_pdf = (shape_k / scale) * (bin_centers / scale)**(shape_k - 1) * \
           np.exp(-(bin_centers / scale)**shape_k)

weib_pmf = weib_pdf / weib_pdf.sum()

noise = np.random.choice(bin_centers, size=n, p=weib_pmf)

########################################################

mu = np.mean(noise)
sigma = np.std(noise)
Z = (noise - mu) / sigma

# this is the normal distribution we're overlaying on top of our histogram
normal_distribution = norm(0, 1)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(Z, bins=40, density=True, alpha=0.5, color='skyblue')
x_std = np.linspace(min(Z), max(Z), 300)
ax.plot(x_std, normal_distribution.pdf(x_std), 'k-', label='Standard Normal')
ax.set_title("Sampled Noise with Normal Distribution Overlayed")

plt.tight_layout()
plt.show()


counts, _ = np.histogram(noise, bins=bins)
emp_p = counts / counts.sum()

# Compare to Weibull
weib_q = weib_pmf 

divergence_weibull = 0.0
for x, y in zip(emp_p, weib_q):
    if x > 0 and y > 0:
        divergence_weibull += x * math.log2(x / y)

print("Divergence (Weibull) =", divergence_weibull)

########################################################
# FOR COMPARISON: calculate divergence between normal and then geometric!
mu_noise = np.mean(noise)
sigma_noise = np.std(noise)
normal_dist = norm(mu_noise, sigma_noise)

normal_pdf = normal_dist.pdf(bin_centers)
normal_pmf = normal_pdf / normal_pdf.sum()

divergence_normal = 0.0
for x, y in zip(emp_p, normal_pmf):
    if x > 0 and y > 0:
        divergence_normal += x * math.log2(x / y)

print("Divergence (Normal) =", divergence_normal)

bin_indices = np.arange(len(bin_centers)) + 1 
mean_bin_index = np.sum(emp_p * bin_indices)
p_geom = 1 / (1 + mean_bin_index)


geom_pmf_raw = geom.pmf(bin_indices, p_geom)
geom_pmf = geom_pmf_raw / geom_pmf_raw.sum()

divergence_geometric = 0.0
for x, y in zip(emp_p, geom_pmf):
    if x > 0 and y > 0:
        divergence_geometric += x * math.log2(x / y)

print("Divergence (Geometric) =", divergence_geometric)