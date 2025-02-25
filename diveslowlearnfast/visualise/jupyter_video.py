import matplotlib.pyplot as plt
import torch
from ipywidgets import HTML
from matplotlib import animation


def vidshow(video, interval, title=None, axis='on'):
    if type(video) == torch.Tensor:
        video = video.cpu().numpy().transpose(0, 2, 3, 1)

    fig = plt.figure()
    plt.ioff()  # Turn off interactive mode
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)

    img = ax.imshow(video[0])
    ax.axis(axis)
    def _update(frame):
        img.set_data(video[frame])

    ani = animation.FuncAnimation(fig, _update, frames=len(video), interval=interval)

    return HTML(ani.to_jshtml()), ani