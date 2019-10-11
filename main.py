import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def load_data(file):
    return sio.loadmat(file)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


if __name__ == "__main__" :

    data = load_data("trainecg.mat")

    diseased = data["diseased"]
    healthy = data["healthy"]

    diseased_mean = np.mean(diseased, axis=0)
    healthy_mean = np.mean(healthy, axis=0)

    diseased_var = np.var(diseased, axis=0)
    healthy_var = np.var(healthy, axis=0)

    auto_dis = autocorr(diseased[1, :])
    auto_hea = autocorr(healthy[1, :])

    print(auto_dis.shape)

    fig, axes = plt.subplots(2, 2, figsize=(24, 8))

    axes[0][0].plot(np.array(range(1,4001)), diseased[1, :], '-o' + 'k', label="Data")
    axes[0][0].set_title("Diseased example")
    axes[0][0].legend()

    axes[0][1].plot(auto_dis / float(auto_dis.max()),
                 '-' + 'k', label="Corr")
    axes[0][1].legend()
    axes[0][1].set_title("Autocorrelation diseased")

    axes[1][0].plot(np.array(range(1,4001)), healthy[1, :], '-o' + 'k', label="Data")
    axes[1][0].set_title("Healthy example")
    axes[1][0].legend()

    axes[1][1].plot(auto_hea / float(auto_hea.max()),
                 '-' + 'k', label="Corr")
    axes[1][1].legend()
    axes[1][1].set_title("Autocorrelation diseased")

    plt.show()


