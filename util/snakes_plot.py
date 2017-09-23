import matplotlib.pyplot as plt


def plot_snakes_as_contours(df):

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(2, 5, 1)
    i = 1

    for title in df.index:

        ax = fig.add_subplot(2, 5, i)
        ax.imshow(df.loc[title]['test_im'])
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        ax = fig.add_subplot(2, 5, i+5)
        ax.imshow(df.loc[title]['test_im'], cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])
        contour = ax.contour(df.loc[title]['snakes'], [0.5], colors='r')
        i += 1
    fig.subplots_adjust(hspace=0)
    fig.show()


def plot_inverse_snakes_as_contours(df):

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(2, 5, 1)
    i = 1

    for title in df.index:

        ax = fig.add_subplot(2, 5, i)
        ax.imshow(df.loc[title]['test_im'])
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        ax = fig.add_subplot(2, 5, i+5)
        ax.imshow(df.loc[title]['test_im'], cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])
        contour = ax.contour(df.loc[title]['inv_snakes'], [0.5], colors='r')
        i += 1
    fig.subplots_adjust(hspace=0)
    fig.show()

