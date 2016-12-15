# Author: Johannes Gätjen
import numpy as np
import scipy.spatial.distance as dst
import math
import arff
import numpy.matlib as ml
import scipy.optimize as opt


# valid metrics: euclidean, seuclidean, sqeuclidean, hamming, cityblock, chebyshev and more
def kde_train(data, labels, bandwidth=1., metric='euclidean', bw_method='manual'):
    numLabels = len(set(labels))
    # already rescale data here when using seuclidean metric, so that we can use silverman and plugin
    data_ = data
    if metric == 'seuclidean':
        data_ = data - ml.repmat(np.min(data, 0), len(labels), 1)
        data_ = data_ / ml.repmat(np.max(data_, 0), len(labels), 1)
    # Silverman's rule of thumb is technically only for univariate data, for which a gaussian distribution is assumed
    # For multivariate data, we simply take the average of the standard deviations of the different dimensions
    # possibly unsuitable when using a metric other than euclidean
    if bw_method == 'silverman':
        bandwidth = []
        # iterate over classes
        for label in range(numLabels):
            labelData = data_[labels == label, :]
            std = np.mean(np.std(labelData, 0))
            bw = ((4 * std ** 5)/(3 * np.shape(labelData)[0])) ** 0.2
            bandwidth.append(bw)
    # solve-the-equation plugin method, as with silverman's rule, average over dimensions, because I'm lazy
    # implemented according to chapter 5.2 of http://www.umiacs.umd.edu/labs/cvl/pirl/vikas/publications/CS-TR-4774.pdf
    # is optimized for gaussian kernel
    # behaviour not so stable, e.g. when some of the values are too small for floating point precision errors will occur
    # due to division through zero. The Newton optimization may also fail when h becomes negative.
    elif bw_method == 'plugin':
        bandwidth = []
        for label in range(numLabels):
            labelData = data_[labels == label, :]
            bw = plugin_estimate(labelData)
            bandwidth.append(bw)
    else:
        if isinstance(bandwidth, float):
            bandwidth = [bandwidth] * numLabels
        elif len(bandwidth) != numLabels:
            bandwidth = [bandwidth[0]] * numLabels
        # else: user has specified separate bandwidths for separate classes, take as is, assumes bandwidth is list
        # should probably do more error checking here
    print('bandwidths:', bandwidth)
    model = {'data': data, 'labels': labels, 'numLabels': numLabels, 'bandwidth': bandwidth}
    return model


def kde_classify(model, data, kernel='gaussian', metric='euclidean'):
    bandwidth = model['bandwidth']
    # assumes labels are always integers increasing in steps of one, with zero being the "first" class label
    labels = []
    # delete dimensions of the data where all instances have the same value, because else seuclidean only produces nan
    if metric == 'seuclidean':
        all_equal_columns = [i for i in range(data.shape[1]) if len(set(data[:, i].flat)) == 1]
        model['data'] = np.delete(model['data'], all_equal_columns, 1)
        data = np.delete(data, all_equal_columns, 1)
    # rows=points to classify, columns=density heights of other points
    distances = dst.cdist(data, model['data'], metric=metric)
    print(metric)
    # compute the respective heights of the kernels at the different points for the data
    density_height = [[kernel_height(pair, bandwidth[model['labels'][idx]], kernel) for idx, pair in enumerate(row)] for row in distances]
    # add the heights for the different classes and get the class with the maximum height
    for row in density_height:
        label_densities = [sum([h for idx, h in enumerate(row) if model['labels'][idx] == l]) for l in range(model['numLabels'])]
        labels.append(label_densities.index(max(label_densities)))
    return labels


def kernel_height(distance, bandwidth, kernel):
    rtn = 0
    dist = distance/bandwidth
    if kernel == 'gaussian':
        rtn = (1/math.sqrt(2*math.pi)) * math.exp(-0.5 * dist ** 2) / bandwidth
    elif kernel == 'epanechnikov':
        rtn = max(0, 3/4*(1 - dist**2)/bandwidth)
    elif kernel == 'uniform':
        if distance < bandwidth:
            rtn = 1/(2*bandwidth)
    elif kernel == 'triangular':
        rtn = max(0, (1-dist)/bandwidth)
    elif kernel == 'quartic':
        if distance < bandwidth:
            rtn = 15/16*(1 - dist**2)**2/bandwidth
    return rtn


def plugin_estimate(samples):
    dims = np.shape(samples)[1]
    total = 0
    for d in range(dims):
        dim_samples = samples[:, d]
        d_h = opt.newton(h_optimize, 50, args=(dim_samples, ), maxiter=15)
        print('dimension', d, d_h)
        total += d_h
    rtn = total / dims
    print('final estimated h:', rtn)
    return rtn


# the function that is optimized, i.e. which we search the root for
# samples: vector with all the samples
def h_optimize(h, samples):
    n = len(samples)
    gamma_h = gamma(h, samples)
    return h - np.power(1/(2*math.sqrt(math.pi)*n*phi_4(gamma_h, samples)), 0.2)


# the phi_4(x) function
def phi_4(x, samples):
    n = len(samples)
    hermite_sum = 0
    for xi in samples:
        for xj in samples:
            hermite_sum += hermite_4((xi - xj)/x)*math.exp(-(xi - xj)**2/(2 * x ** 2))
    return hermite_sum / (n * (n-1) * math.sqrt(2 * math.pi) * x ** 5)


# the phi_6(x) function
def phi_6(x, samples):
    n = len(samples)
    hermite_sum = 0
    for xi in samples:
        for xj in samples:
            hermite_sum += hermite_6((xi - xj)/x)*math.exp(-(xi - xj)**2/(2 * x ** 2))
    return hermite_sum / (n * (n-1) * math.sqrt(2 * math.pi) * x ** 7)


# the sixth probabilists' hermite polynomial
def hermite_6(x):
    return x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15


# the fourth probabilists' hermite polynomial
def hermite_4(x):
    return x ** 4 - 6 * x ** 2 + 3


def gamma(h, samples):
    sigma = np.std(samples)
    n = len(samples)
    phi_6_est = (-15/(16*math.sqrt(math.pi))) * sigma ** -7
    phi_8 = (105/(32*math.sqrt(math.pi))) * sigma ** -9
    g1 = (-6/(math.sqrt(2 * math.pi) * n * phi_6_est)) ** (1/7)
    den = (math.sqrt(2 * math.pi) * n * phi_8)
    g2 = (30/den)
    g2 **= (1/9)
    rtn = (-6 * math.sqrt(2) * phi_4(g1, samples)/phi_6(g2, samples))
    print('rtn:', rtn, 'h:', h)
    rtn = np.power(rtn, (1/7)) * np.power(h, 5/7)
    return rtn[0, 0]


def digit_transform(labels):
    for idx, l in enumerate(labels):
        if l in [0, 2, 4, 6, 8]:
            labels[idx] = l+1
        else:
            labels[idx] = l-1
    return labels


if __name__ == '__main__':
    exa = 'iris'
    met = 'seuclidean'
    krnl = 'gaussian'
    training = arff.load(open('./data/' + exa + '_tr.arff'), 'rb')
    tr_data = np.matrix([x[:-1] for x in training['data']])

    tr_labels = np.array([x[-1] for x in training['data']])
    # md = kde_train(tr_data, tr_labels, bw_method='silverman', metric=met)  # bandwidth=0.3
    md = kde_train(tr_data, tr_labels, bandwidth=0.21, metric=met)  # bandwidth=0.3
    test = arff.load(open('./data/' + exa + '_te.arff'), 'rb')
    te_data = np.matrix([x[:-1] for x in test['data']])

    te_labels = [x[-1] for x in test['data']]

    results = kde_classify(md, te_data, metric=met, kernel=krnl)
    # results = digit_transform(results)
    print('classes:', md['numLabels'])
    print(results)
    print(te_labels)
    print('for binary classifiers, ratio of positive labels:', sum(te_labels)/len(te_labels))
    num_wrong = sum([rs != te_labels[idx] for idx, rs in enumerate(results)])
    print(num_wrong, (1 - num_wrong/len(results))*100, '%')

