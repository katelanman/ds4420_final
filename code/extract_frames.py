import cv2
import pyarrow as pa
import pyarrow.feather as feather
import numpy as np
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

def extract_frames(filename, per_secs, label):
    vidcap = cv2.VideoCapture(filename)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()

    frames = []
    i = 0
    while success:
        # save only frames every per_secs
        if i % (per_secs * fps) == 0:
            # resize image to be (288, 352), remove color dimension, and flatten
            image = cv2.resize(image,(288,352))[:,:,0].flatten()

            frames.append(np.append(image, label)) 

        success,image = vidcap.read()
        i += 1

    return frames

if __name__ == "__main__":
    fish_frames = extract_frames('data/working/fish_frames.mp4', 2, 1)
    print("fist frames done")

    no_fish_frames = extract_frames('data/working/no_fish_frames.mp4', 2, 0)
    print("no fish frames done")

    columns = list(range(101376))
    columns.append("label")

    frames = pd.DataFrame(fish_frames + no_fish_frames, columns=columns)
    print(frames)

    feather.write_feather(frames, "data/working/fish_frames.feather")