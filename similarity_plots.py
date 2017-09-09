import matplotlib.pyplot as plt


def generate_plots(depth, test):

    fig = plt.figure()
    ax1 = fig.add_subplot(112)
    ax1.imshow(depth)
    ax1.set_title('Normalized Snow Depth')
    plt.colorbar()

    ax2 = fig.add_subplot(122)
    ax2.imshow(test)
    ax2.set_title('test')
    plt.colorbar()
