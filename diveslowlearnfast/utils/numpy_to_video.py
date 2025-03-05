import cv2

def numpy_to_video(frames, output_filename, fps=30, codec='mp4v'):
    """
    Convert a numpy array of frames to an MP4 video file.

    Parameters:
    frames (numpy.ndarray): Array of shape (n_frames, height, width, channels)
                           where channels is typically 3 for RGB
    output_filename (str): Path to save the output video
    fps (int): Frames per second
    codec (str): FourCC codec code (default: mp4v for MP4)
    """
    # Get dimensions of the frames
    n_frames, height, width, channels = frames.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for i in range(n_frames):
        if channels == 3:
            frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        else:
            frame = frames[i]

        out.write(frame)

    out.release()