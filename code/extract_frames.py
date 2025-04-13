import cv2
import pyarrow as pa
import pyarrow.feather as feather
import numpy as np
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

def detect_and_remove_letterboxing(image):
    """Detect and remove letterboxing (black bars) from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Get the sum of pixel values for each row to detect black bars
    row_sums = np.sum(gray, axis=1)
    
    # Find the first and last rows that aren't completely (or mostly) black
    # Use a small threshold to account for compression artifacts
    threshold = np.mean(row_sums) * 0.1
    non_black_rows = np.where(row_sums > threshold)[0]
    
    if len(non_black_rows) > 0:
        top_crop = non_black_rows[0]
        bottom_crop = non_black_rows[-1] + 1
        
        # Only crop if black bars are detected
        if top_crop > 0 or bottom_crop < image.shape[0]:
            return image[top_crop:bottom_crop, :]
    
    # Return original if no letterboxing detected
    return image

def extract_frames(filename, per_secs, label):
    vidcap = cv2.VideoCapture(filename)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()

    frames = []
    i = 0
    while success:
        # save only frames every per_secs
        if i % (per_secs * fps) == 0:
            # Remove letterboxing if present
            image = detect_and_remove_letterboxing(image)
            
            # resize image to be (352, 288), remove color dimension, and flatten
            image = cv2.resize(image, (352, 288))[:,:,0].flatten()

            frames.append(np.append(image, label)) 

        success, image = vidcap.read()
        i += 1

    return frames

if __name__ == "__main__":
    fish_frames = extract_frames('data/working/fish_frames.mp4', 1, 1)
    print("fish frames done")
    print(len(fish_frames))
    no_fish_frames = extract_frames('data/working/no_fish_frames.mp4', 4, 0)
    np.savetxt("no_fish_frames.csv", no_fish_frames, delimiter=",")
    print("no fish frames done")
    print(len(no_fish_frames))

    columns = list(range(101376))
    columns.append("label")
    
    frames = pd.DataFrame(fish_frames + no_fish_frames, columns=columns)
    print(frames)
    feather.write_feather(frames, "data/working/fish_frames.feather")