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
            # remove color dimension and flatten
            flattened = list(image[:,:,0].flatten())
            print(label, image.shape)
            frames.append(np.append(flattened, label)) 
            break

        success,image = vidcap.read()
        i += 1

    return frames

fish_frames = extract_frames('data/working/labelled_videos/fish_frames.mp4', 2, 1)
no_fish_frames = extract_frames('data/working/labelled_videos/no_fish_frames.mp4', 2, 0)

# frames = pd.DataFrame(fish_frames + no_fish_frames, columns=list(range(101376)).append("label"))
# print(frames)
# feather.write_feather(pa.Table.from_arrays(frames, names=names), "data/working/fish_frames.feather")