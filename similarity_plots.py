import matplotlib.pyplot as plt


def generate_plots(comparison):

    plt.figure()
    plt.imshow(comparison.mse_map)
    plt.suptitle(comparison.cname)
    plt.title('MSE: ' + str(comparison.mse))
    plt.xlabel('m')
    plt.ylabel('m')
    plt.colorbar()
    plt.savefig(comparison.results_dir + '/mse_map.png', dpi=300)