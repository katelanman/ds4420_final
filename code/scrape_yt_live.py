from yt_dlp import YoutubeDL

SAVEPATH = "data/original/"

## Downloading a YouTube Video
def download_video(video_url, **kwargs):
    opts = {**kwargs}
    with YoutubeDL(opts) as yt:
        yt.download([video_url])
        print(f"Downloaded video: {video_url}")

# Download a video
video_url =  "https://www.youtube.com/watch?v=JOEdSqXEZnk"
download_video(video_url, paths={"home":SAVEPATH}, live_from_start=True, nopart=True)