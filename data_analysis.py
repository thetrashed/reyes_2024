import numpy as np
import matplotlib.pyplot as plt
import matplotlib


import glob
import os

# Plateau fit linewidth
linewidth = 0.5

# Data/Plot directories
data_dirs = ["pion_P2/", "pion_P4/"]
plots_dir = "plots/"

# Max time to include in the plot
max_times = [20, 20]

# Plateau fit time range
index_t_lows = [4, 4]
index_t_highs = [8, 7]

# y-axis limit
y_lims_low = [0.2, 0.0]
y_lims_high = [0.5, 1.0]

# Lattice constant
a = (2 * np.pi) / 32

# Pion mass
m = 0.125


def plot_dispersion_relation(E_plat, momenta):
    ps = np.linspace(0, np.ceil(max(momenta)), 1000)
    Es = np.sqrt(np.add(m**2, np.multiply((a**2), np.square(ps))))
    print(E_plat)

    E_plats = [E_plat[i][0] for i in range(len(E_plat))]
    E_plat_errs = [E_plat[i][1] for i in range(len(E_plat))]
    fig = plt.figure()
    plt.plot(ps, Es, color="black", label="Theoretical")
    plt.errorbar(momenta, E_plats, yerr=E_plat_errs, fmt="x", label="Experimental")
    plt.xlabel(r"$p$")
    plt.ylabel(r"$aE_{plat}$")
    plt.legend()
    fig.savefig(plots_dir + "dispersion_relation.png", dpi=300)
    plt.close()


# Takes in a 1-d array and returns the jackknife resampled result of the array
def jackknife_sampling(A):
    resampled = np.empty(len(A))
    for i in range(len(A)):
        resampled[i] = np.mean(np.delete(A, i))

    return resampled


def jackknife_error(Q):
    # return np.multiply(np.std(Q, ddof=1), np.sqrt(np.square(len(Q) - 1) / len(Q)))
    return np.multiply(np.std(Q), np.sqrt(len(Q) - 1))


def read_data_file(fname):
    with open(fname, "r") as f:
        data = fp.read().split("\n")
        data = [d.split(" ") for d in data if d != ""]

    return np.array([d[4] for d in data])


def plateau_fitting(energies, errors, start_time, end_time):
    energies = np.array(energies)

    E_disp = []

    tmp = np.square(errors[start_time:end_time])
    denominator = np.sum(1 / tmp)
    for energy_bin in energies.T:
        numerator = np.sum(energy_bin[start_time:end_time] / tmp)

        E_disp.append(numerator / denominator)

    return E_disp


def analyze_data(
    data,
    plots_dir,
    momentum,
    fname,
    max_time,
    index_t_low,
    index_t_high,
    y_lim_low=None,
    y_lim_high=None,
):
    data = np.array(data).T

    # Average the values for the 2pt function for t_1 = t and t_2 = 63 - t
    averaged_data = np.divide(
        np.add(
            data[1 : data.shape[0] // 2],
            np.fliplr(data[(data.shape[0] // 2) + 1 :]),
        ),
        2,
    )
    # averaged_data = []
    # for i in range(data.shape[0] // 2):
    # averaged_data.append(np.divide(np.add(data[i], data[data.shape[0] - 1 - i]), 2))

    # Get jackknife bins for the data
    samples = []
    for d in averaged_data:
        samples.append(jackknife_sampling(d))

    # Central values of the ratios of the 2pt functions and also calculate the jackknife errors
    energies = []
    errors = []
    j = 0
    for i in range(1, max_time):
        tmp = np.divide(samples[i], samples[i + 1])
        energies.append(np.log(tmp))
        errors.append(jackknife_error(energies[j]))
        j += 1

    central_vals = [np.mean(energy) for energy in energies]

    # Plateau fitting
    E_disp = plateau_fitting(energies, errors, index_t_low, index_t_high)
    jackknife_E_disp = jackknife_sampling(E_disp)

    E_disp_error = jackknife_error(jackknife_E_disp)

    # Central values for E_disp
    cv_E_disp = [np.mean(jackknife_E_disp)] * (index_t_high - index_t_low)

    # Plot the energy using matplotlib
    ts = [i + 1 for i in range(len(central_vals))]
    fig = plt.figure()
    plt.scatter(ts, central_vals, marker="x", label=r"$E_{eff}$", linewidth=linewidth)
    plt.errorbar(ts, central_vals, yerr=errors, fmt="x", linewidth=linewidth)

    plt.plot(
        ts[index_t_low:index_t_high],
        cv_E_disp,
        color="black",
        label="Plateau fit",
        linewidth=linewidth,
    )
    plt.fill_between(
        ts[index_t_low:index_t_high],
        cv_E_disp - E_disp_error,
        cv_E_disp + E_disp_error,
        alpha=1,
        color="red",
    )

    plt.ylabel(r"$E_{eff}$")
    plt.xlabel(r"$t$")
    if not (y_lim_low is None and y_lim_high is None):
        plt.ylim((y_lim_low, y_lim_high))
    plt.legend()
    # plt.show()
    fig.savefig(
        plots_dir
        + fname
        + "-"
        + np.format_float_positional(momentum, min_digits=3, precision=3)
        + ".png",
        dpi=300,
    )
    plt.close()

    # Return the plateau value and the error in the plateau
    return cv_E_disp[0], E_disp_error


E_plat = []
momenta = []
# Get a list of the sub-directories
for j, data_dir in enumerate(data_dirs):
    data = {}
    dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    # Loop over the files in each sub-directory (and average the two point functions)
    # c2pt_data = [[] for i in range(64)]

    for dir in dirs:
        files = glob.glob("*.dat", root_dir=dir, recursive=True)
        if files == []:
            continue

        with open(dir + "/" + files[0], "r") as f1, open(
            dir + "/" + files[1], "r"
        ) as f2:
            tmp1 = f1.read().split("\n")
            tmp2 = f2.read().split("\n")

            prev_momentum = 0
            c2pts = [[] for _ in range(64)]
            for point in zip(tmp1, tmp2):
                if point == ("", ""):
                    if not (prev_momentum in data):
                        data[prev_momentum] = []

                    data[prev_momentum].append(np.mean(c2pts, axis=1))
                    c2pts = [[] for i in range(64)]
                    prev_momentum = momentum
                    break

                d1 = point[0].split(" ")
                d2 = point[1].split(" ")
                momentum = np.sqrt(int(d1[0]) ** 2 + int(d1[1]) ** 2 + int(d1[2]) ** 2)
                t = int(d1[3])

                if prev_momentum == 0:
                    prev_momentum = momentum

                if momentum != prev_momentum:
                    if not (prev_momentum in data):
                        data[prev_momentum] = []

                    data[prev_momentum].append(np.mean(c2pts, axis=1))
                    c2pts = [[] for i in range(64)]
                    prev_momentum = momentum

                c2pts[t].append((float(d1[4]) + float(d2[4])) / 2)

    momenta.extend(data.keys())
    for momentum, c2pt_data in data.items():
        E_plat.append(
            analyze_data(
                c2pt_data,
                plots_dir,
                momentum,
                data_dir[:-1],
                max_times[j],
                index_t_lows[j],
                index_t_highs[j],
                # y_lims_low[j],
                # y_lims_high[j],
            )
        )

plot_dispersion_relation(E_plat, momenta)
