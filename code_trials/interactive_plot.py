import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import cv2

def interactive_image_and_intensity(image, axis=0):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n_slices = image.shape[axis]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    idx = 0
    if axis == 0:
        img_slice = image[idx, :, :]
        intensity = grey_image[idx, :]
        pixels = np.arange(len(intensity))
        ref_line = axs[0].axhline(y=idx, color='red', linestyle='--')
    elif axis == 1:
        img_slice = image[:, idx, :]
        intensity = grey_image[:, idx]
        pixels = np.arange(len(intensity))
        ref_line = axs[0].axvline(x=idx, color='red', linestyle='--')
    else:
        img_slice = image[:, :, idx]
        intensity = img_slice.sum(axis=0)
        pixels = np.arange(img_slice.shape[1])
        ref_line = None

    axs[0].imshow(image)#im = axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title(f'Image Slice (axis={axis}, idx={idx})')
    axs[0].grid()
    line, = axs[1].plot(pixels, intensity)
    axs[1].set_title('Intensity vs Pixel')
    axs[1].set_xlabel('Pixel')
    axs[1].set_ylabel('Intensity')
    axs[1].grid()

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, n_slices-1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        if axis == 0:
            img_slice = image[idx, :, :]
            intensity = grey_image[idx, :]
            pixels = np.arange(len(intensity))
            ref_line.set_ydata([idx,idx])
        elif axis == 1:
            img_slice = image[:, idx, :]
            intensity = grey_image[:, idx]
            pixels = np.arange(len(intensity))
            ref_line.set_xdata([idx,idx])
        else:
            img_slice = image[:, :, idx]
            intensity = img_slice.sum(axis=0)
            pixels = np.arange(img_slice.shape[1])
        axs[0].imshow(image)#im.set_data(img_slice)
        axs[0].set_title(f'Image Slice (axis={axis}, idx={idx})')
        line.set_data(pixels, intensity)
        axs[1].relim()
        axs[1].autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

image_name = "C:\\Users\\Jason.Bane\\Documents\\Nautilus\\DOE_stage\\code_trials\\images\\Image00000.BMP"
image = cv2.imread(image_name)
interactive_image_and_intensity(image, axis=1)