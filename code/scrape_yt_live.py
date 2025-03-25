from yt_dlp import YoutubeDL

SAVEPATH = "data/original/"

## Downloading a YouTube Video
def download_video(video_urls, **kwargs):
    opts = {**kwargs}
    with YoutubeDL(opts) as yt:
        yt.download(video_urls)

# Download livestream
# live =  "https://www.youtube.com/watch?v=JOEdSqXEZnk"
# download_video([live], paths={"home":SAVEPATH}, live_from_start=True, nopart=True)

# Download highlights
highlights =  ["https://www.youtube.com/watch?v=fNLWiOhBklE",
               "https://www.youtube.com/watch?v=EOmnIjdWzOE"]
download_video(highlights, paths={"home":SAVEPATH+"to_label/"}, nopart=True)