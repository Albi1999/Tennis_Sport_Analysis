# Used essentially to gather data for different tennis game scenarios to show robustness of our models 

# NEEDED LIBRARIES : 
# pip install yt-dlp
# brew install ffmpeg

import subprocess

def download_clip(url, start, end):
    """
    Download YouTube video segment without audio.
    start and end should be in format like '00:30' or '01:30'
    """
    command = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--format", "bv*[height<=720]",
        "-S vcodec:h264,res,acodec:m4a",
        url
    ]
    subprocess.run(command)




def main():
    url = 'https://www.youtube.com/watch?v=5uFAkizQNJI&ab_channel=Wimbledon'
    start = '00:00:00'
    end = '00:00:08'
    download_clip(url, start, end)


if __name__ == '__main__':
    main()


