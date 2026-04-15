import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fit_RV = [2.389, 2.492, 2.749, 2.814, 2.824, 2.828, 2.855, 2.855, 2.904, 2.962, 3.003, 3.023, 3.107, 5.662, 5.346]
    est_RV = [3.496, 3.248, 3.063, 3.114, 3.644, 3.248, 3.358, 3.323, 3.484, 3.491, 3.571, 3.342, 3.683, 6.241, 5.767]
    mean_diff = np.mean(abs(np.array(fit_RV) - np.array(est_RV))/np.array(fit_RV))
    est_RV = (1.-mean_diff)*np.array(est_RV)
    
    x = [2.24, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 6.8]
    y = [2.24, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 6.8]
    plt.plot(x,y, '--')
    plt.scatter(est_RV, fit_RV)
    plt.xlim(2.24,6.8)
    plt.ylim(2.24,6.8)
    plt.ylabel('fitted R(V)')
    plt.xlabel('estimated R(V)')
    plt.show()