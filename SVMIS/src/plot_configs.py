import matplotlib.pyplot as plt

# sizes for single row figure
w_inc_fullpage, h_inc = 7.08, 2.

def set_plot_configs():
    # set defaults for plots
    # font size
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['legend.title_fontsize'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    # line width
    plt.rcParams['lines.linewidth'] = 0.8

    # set splines line width
    plt.rcParams['axes.linewidth'] = 0.6

    # marker size
    plt.rcParams['lines.markersize'] = 6

    # set dpi
    plt.rcParams['figure.dpi'] = 200



    

    