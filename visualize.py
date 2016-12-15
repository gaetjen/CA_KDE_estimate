# generating some plots, to show the motivation for kde vs. histograms
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

pic_name = 0

x = np.linspace(0, 10, 201)
gauss1_mu = 3
gauss1_sigma = 0.7
gauss1_h = 2
gauss2_mu = 6.8
gauss2_sigma = 1.8
gauss2_h = 3

gauss1 = norm.pdf(x, gauss1_mu, gauss1_sigma) * gauss1_h
gauss2 = norm.pdf(x, gauss2_mu, gauss2_sigma) * gauss2_h

pdf = gauss1 + gauss2
pdf /= sum(pdf)
pdf /= x[1]
cdf = np.cumsum(pdf) * x[1]
# cdf /= cdf[-1]
plt.plot(x, pdf)
# print(np.mean(pdf))
# plt.show()
# plt.plot(x, cdf)
# plt.show()

num_samples = 20
samples = []
random.seed(1337)
for _ in range(num_samples):
    r = random.random()
    x_value = x[np.argmax(cdf > r)-1]
    samples.append(x_value)

plothandle = plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
plt.xlabel('x')
plt.ylabel('density')
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
plt.close()

for bins in range(1, 21):
    plt.figure()
    plt.plot(x, pdf)
    plothandle = plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
    plt.xlabel('x')
    plt.ylabel('density')
    plt.hist(samples, bins=bins, normed=True)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 0.4])
    plt.title('number of bins='+str(bins))
    plt.savefig('./pics/'+str(pic_name)+'.png')
    pic_name += 1
    plt.close()

num_bins = 10
for offset in range(-3, 9):
    plt.figure()
    plt.plot(x, pdf)
    plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
    plt.xlabel('x')
    plt.ylabel('density')
    bins = np.array([i for i in range(num_bins+1)]) / (num_bins+1)*x[-1]
    bins += offset/bins[1]/11
    plt.hist(samples, bins=bins, normed=True)
    # plt.hist()
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 0.4])
    plt.title('offset='+str(offset/bins[1]/11))
    plt.savefig('./pics/'+str(pic_name)+'.png')
    pic_name += 1
    plt.close()
# plt.hist(samples, bins=5, normed=True)
# plt.show()

plt.figure()
plt.plot(x, pdf)
plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
plt.xlabel('x')
plt.ylabel('density')
kde = x * 0
for s in samples:
    sample_rect = x*0
    sample_rect[abs(x-s) < 0.6] = 1/num_samples/1.2
    kde += sample_rect
    plt.plot(x, sample_rect, ls='--', lw=0.5, color='black')
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
plt.plot(x, kde)
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
# plt.show()
plt.close()

plt.figure()
plt.plot(x, pdf)
plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
plt.xlabel('x')
plt.ylabel('density')
kde = x * 0
bw = 0.8
for s in samples:
    sample_gauss = norm.pdf(x, s, bw)/num_samples
    kde += sample_gauss
    plt.plot(x, sample_gauss, ls='--', lw=0.5, color='black')

plt.plot(x, kde)
# plt.show()
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
plt.close()

plt.figure()
c1 = norm.pdf(x, 2, 0.5) + norm.pdf(x, 4, 1)*0.5
c2 = norm.pdf(x, 7, 2.5) + norm.pdf(x, 6, 0.3)*0.5
c1 /= sum(c1)*x[1]
c2 /= sum(c2)*x[1]
plt.plot(x, c1)
plt.plot(x, c2)
plt.xlabel('x')
plt.ylabel('density')
axes = plt.gca()
axes.set_xlim([0, 10])
plt.title('classification')
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
plt.close()

bw = 0.3
plt.figure()
plt.plot(x, pdf)
plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
plt.xlabel('x')
plt.ylabel('density')
bw /= 10
kde = x * 0
for s in samples:
    sample_gauss = norm.pdf(x, s, bw)/num_samples
    kde += sample_gauss
plt.plot(x, kde)
axes = plt.gca()
axes.set_xlim([0, 10])
axes.set_ylim([0, 0.4])
plt.title('bandwidth='+str(bw))
plt.savefig('./pics/'+str(pic_name)+'.png')
pic_name += 1
plt.close()

for bw in range(1, 10):
    plt.figure()
    plt.plot(x, pdf)
    plt.plot(samples, [0]*num_samples, marker='x', ls='', markersize=10, mew=2)
    plt.xlabel('x')
    plt.ylabel('density')
    bw /= 10
    kde = x * 0
    for s in samples:
        sample_gauss = norm.pdf(x, s, bw)/num_samples
        kde += sample_gauss
    plt.plot(x, kde)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 0.4])
    plt.title('bandwidth='+str(bw))
    plt.savefig('./pics/'+str(pic_name)+'.png')
    pic_name += 1
    plt.close()


