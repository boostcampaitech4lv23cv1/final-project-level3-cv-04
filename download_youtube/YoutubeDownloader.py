import argparse
from pytube import YouTube
import os
import os.path as osp
from datetime import date, datetime, timezone, timedelta
import json
import ffmpeg
import glob
from moviepy.editor import *
import shutil



def get_current_day_time():
    KST = timezone(timedelta(hours=9))
    time_record = datetime.now(KST)
    _day = str(time_record)[:10]
    _time = str(time_record.time())[:5]
    current_time = "".join(_day.split("-")) + "_"+"".join(_time.split(":"))
    return current_time


def save_meta_info(dict_type_info, file_path):
    file_path = file_path.replace(".mp4",".json")
    dir_path, file_name = osp.split(osp.abspath(file_path))
    json.dump(dict_type_info, open(file_path, 'w'), indent=4, ensure_ascii=False)
    print(f"meta-info saved in [{dir_path}], file name is {file_name}")
    return None


def download_audio(yt:YouTube,  output_path: str, filename:str) -> None:
    for stream in yt.streams.filter(only_audio=True).order_by("abr").desc():
        if "mp4a" in stream.audio_codec:
            stream.download(output_path = output_path, filename = f"{filename.split('.')[0]}_audio.mp4")


def clip_audio(meta_info:dict, start_sec:int=0, end_sec:int=60):
    print('audio clip')
    audio_path = meta_info['audio_root']
    audio = AudioFileClip(audio_path).subclip(start_sec, end_sec)
    audio.write_audiofile('./temp_audio.mp3')
    # delete original mp4 audio
    os.remove(audio_path)
    
    audio_filename = osp.basename(audio_path).split('.')[0] + '.mp3' # file name change to mp3
    audio_dir = osp.dirname(audio_path)
    audio_file_relpath = osp.relpath(osp.join(audio_dir, audio_filename))

    # move mp3 file rename
    shutil.move('./temp_audio.mp3', audio_file_relpath)
    meta_info['audio_root'] = './'+audio_file_relpath

    return None

def download_and_capture(youtube_url:int, start_sec:int, end_sec:int, save_dir:str):
    print(f"download: {youtube_url}")
    
    youtube_id = youtube_url.split('=')[-1]
    
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(type="video", file_extension="mp4").order_by("resolution").desc()[0]
    # make meta data, (ex) save dir is 0lXwMdnpoFQ, if you want currenttile file name use --> get_current_day_time()
    meta_info = {}
    meta_info["filename"] = youtube_id+".mp4" # -> 0lXwMdnpoFQ.mp4
    meta_info["title"] = stream.title
    meta_info["description"] = yt.description # yt.length
    meta_info["video_full_length"] = yt.length
    meta_info["vcodec"] = stream.video_codec
    meta_info["fps"] = stream.fps
    meta_info["itag"] = stream.itag
    meta_info["type "] = stream.type
    meta_info["resolution"] = stream.resolution
    meta_info["length"] = yt.length
    meta_info["image_root"] = osp.join(save_dir, 'captures')
    meta_info["audio_root"] = osp.join(save_dir, f"{youtube_id}_audio.mp4")

    ## download video by pytube
    stream.download(output_path = save_dir, filename=meta_info["filename"]) # download video
    print('video download finish')

    ## download audio by pytube, audio is mp4 extension
    download_audio(yt, save_dir, meta_info["filename"]) # download audio
    print('audio download finish')

    ## clip audio
    clip_audio(meta_info, start_sec, end_sec)

    ## [width, height] checked by ffmpeg-python
    file_path = osp.join(save_dir, meta_info["filename"]) # -> 
    probe = ffmpeg.probe(file_path)
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    ## add resolution info to meta
    meta_info["width"] = int(video['width'])
    meta_info["height"] = int(video['height'])
    save_meta_info(meta_info, osp.abspath(file_path)) # call meta info
    print("download path is", file_path)

    ## for capture all vod's frame, we use ffmpeg cli
    img_capture_dir_path = osp.join(save_dir, 'captures')
    os.makedirs(img_capture_dir_path, exist_ok=True)

    ## change timestamp format 
    start_timestamp = str(timedelta(seconds=start_sec))
    end_timestamp = str(timedelta(seconds=end_sec))

    os.system("ffmpeg " + 
    f"-i {file_path} " +
            f"-ss {start_timestamp} -to {end_timestamp} " + # if you want slice videos input -t <sec> command -t 60 
                f"-r {str(meta_info['fps'])} " +
                    "-f image2 " + img_capture_dir_path + "/%d.jpg")

    ## change format 1.jpg → 000001.jpg
    for file in os.listdir(img_capture_dir_path):
        os.rename(osp.join(img_capture_dir_path, file), osp.join(img_capture_dir_path, file.zfill(10)))
        
    return meta_info


def get_args_parser():
    parser = argparse.ArgumentParser('Hello world!', add_help=False)
    parser.add_argument('--youtube_url', default="https://www.youtube.com/watch?v=0lXwMdnpoFQ", type=str)
    parser.add_argument('--download_path', default="/opt/ml/final-project-level3-cv-04/data", type=str)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('download from Youtube', parents=[get_args_parser()])
    args = parser.parse_args()
    for arg in vars(args):
        print("--"+arg, getattr(args, arg))
    download_and_capture(args.youtube_url, args.download_path)