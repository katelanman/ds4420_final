import zipfile
import tempfile
import cv2

"""zf = zipfile.ZipFile('data/original/livestream.mp4.zip')

with tempfile.TemporaryDirectory() as tempdir:
    zf.extractall(tempdir)
    vidcap = cv2.VideoCapture(tempdir + '/livestream.mp4')
    success,image = vidcap.read()
    count = 0
    while count < 1:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1"""

vidcap = cv2.VideoCapture('data/original/livestream.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
success,image = vidcap.read()

count = 0
i = 0
while success:
    if i % (2 * fps) == 0:
        cv2.imwrite("data/original/stream_frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
        count += 1

    success,image = vidcap.read()
    i += 1