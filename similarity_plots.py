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

    plt.figure()
    plt.imshow(comparison.ssim_map)
    plt.suptitle(comparison.cname)
    plt.title('Structural Similarity Index (SSIM): ' + str(comparison.ssim))
    plt.xlabel('m')
    plt.ylabel('m')
    plt.colorbar()
    plt.savefig(comparison.results_dir + '/ssim_map.png', dpi=300)

    plt.figure()
    plt.imshow(comparison.gms_map)
    plt.suptitle(comparison.cname)
    plt.title('Gradient Magnitude Similarity: ' + str(comparison.gms_index))
    plt.xlabel('m')
    plt.ylabel('m')
    plt.colorbar()
    plt.savefig(comparison.results_dir + '/gms_map.png', dpi=300)

