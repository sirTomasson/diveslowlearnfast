import os
import base64

from diveslowlearnfast.utils.numpy_to_video import numpy_to_video

from ipywidgets import HTML
from IPython.display import display


def create_temp_mp4(video, fps, codec):
    temp_video_path = os.path.join('data', f'__TEMP__.mp4')
    numpy_to_video(video, temp_video_path, fps, codec)
    return temp_video_path

def vidshow(video, fps=10, codec='avc1', size=None):
    """
    Displays a video in a HTML video frame.
    Parameters:
        video (numpy.ndarray): video of frames, heigh, width, channels
        fps (int): frames per second
        codec (str): video codec one of 'avc1' or 'mp4v'
        size ((int, int)): video size in (width, height)
    """
    if size is None:
        _, H, W, _ = video.shape
    else:
        H, W = size
    video_path = create_temp_mp4(video, fps, codec)
    assert os.path.exists(video_path)
    video_path = os.path.join(os.getcwd(), video_path)
    """Display video in Jupyter notebook using base64 encoding"""
    with open(video_path, "rb") as file:
        video_base64 = base64.b64encode(file.read()).decode("utf-8")

    video_html = f"""
    <video width="{W}" height="{H}" controls autoplay loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    display(HTML(video_html))