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
 #   command = [
 #       "yt-dlp",
 #       "--download-sections", f"*{start}-{end}",
 #       "--format", "b[height<=720]",
 #       "-S vcodec:h264,res,acodec:m4a",
 #       url
 #   ]


#01:07:18 01:07:33

    command = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--format", "bestvideo+bestaudio/best",
        "-S vcodec:h264,res,acodec:m4a",
        url
    ]
    subprocess.run(command)


# ALL DJOKOVIC AGAINST ALCARAZ
# VIDEOS : 
# https://www.youtube.com/watch?v=ATI9B7ZLof8&t=6127s&ab_channel=AustralianOpenTV # blue court
# https://www.youtube.com/watch?v=2L06HxxP1jM&t=4978s&ab_channel=Wimbledon # green court 
# https://www.youtube.com/watch?v=najrNDwxUrI&ab_channel=Roland-Garros # clay court



def main():
    url = 'https://www.youtube.com/watch?v=ATI9B7ZLof8&t=6127s&ab_channel=AustralianOpenTV'
    start = '01:20:00'     #*! First 20 munites already scraped so we can start from 20th minute (the videos are in the data folder)
    end = '02:00:00'
    download_clip(url, start, end)

if __name__ == '__main__':
    main()