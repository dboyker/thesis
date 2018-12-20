import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def plot_results(encoder, decoder,
                 x_test,
                 y_test,
                 batch_size=128,
                 model_name="vae_mnist"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    scatter1 = ax1.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    fig.colorbar(scatter1, ax=ax1)
    # display a 15x15 2D manifold of digits
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    ax2.imshow(figure, cmap='Greys_r')
    ax2.grid(False)
    plt.savefig(filename)
    plt.show()