import numpy as np
import properscoring as ps


def calc_crps(ts_ori, ts_means, ts_stds):
    crpss = np.zeros_like(ts_means)
    for i in range(crpss.shape[0]):
        for j in range(crpss.shape[1]):
            crpss[i, j] = ps.crps_gaussian(ts_ori[i, j], mu=ts_means[i, j], sig=ts_stds[i, j])
    return crpss


def RMSE(ts_ori, ts_means):
    return np.sqrt(np.power(abs(ts_means - ts_ori), 2).mean(axis=0))
