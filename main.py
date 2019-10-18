import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import MAP as map
import scipy.stats

def load_data(file):
    return sio.loadmat(file)


def single_autocorrelation(scan):
    result = np.correlate(scan, scan, mode='full')
    autocorr = result[result.size // 2:]
    return autocorr / float(autocorr.max())


def my_single_autocorrelation(scan):
    l = len(scan)
    a = np.zeros(l)
    for k in range(0, l):
        offset = l - k
        a[k] = scan[0: offset - 1].dot(scan[k: l - 1])
        a[k] = a[k] / float(l)
    return a / float(a.max())


def calculate_autocorr(scans):
    a = np.zeros(shape=(scans.shape[0], scans.shape[1]))
    for i in range(0, scans.shape[0]):
        a[i] = my_single_autocorrelation(scans[i, :])
    return a


if __name__ == "__main__":

    data = load_data("data/trainecg.mat")

    diseased = data["diseased"]
    healthy = data["healthy"]

    num_dis = diseased.shape[0]
    num_health = healthy.shape[0]
    num_points = diseased.shape[1]

    auto_dis = calculate_autocorr(diseased)
    auto_hea = calculate_autocorr(healthy)

    x_axis = range(0, 4000)

    fig, axes = plt.subplots(2, 1, figsize=(24, 8))

    for i in range(num_health):
        axes[0].plot(x_axis, auto_hea[i], '-' + 'b')

    for i in range(num_dis):
        axes[1].plot(x_axis, auto_dis[i], '-' + 'r')

    k_star = 2950
    # data_for_k = np.concatenate((auto_hea[:, k], auto_dis[:, k]))
    # fig1, ax1 = plt.subplots()
    #
    # ax1.plot(auto_dis[:, k], '*' + 'r')
    # ax1.plot(auto_hea[:, k], '*' + 'b')

    plt.show()

    dis_mean = np.mean(auto_dis[:,k_star])
    hea_mean = np.mean(auto_hea[:,k_star])

    dis_std = np.std(auto_dis[:,k_star])
    hea_std = np.std(auto_hea[:,k_star])

    P_H, P_D = map.calculate_apriori(num_health, num_dis)

    ProbR_diseased = scipy.stats.norm(dis_mean, dis_std)
    ProbR_healthy = scipy.stats.norm(hea_mean, hea_std)




