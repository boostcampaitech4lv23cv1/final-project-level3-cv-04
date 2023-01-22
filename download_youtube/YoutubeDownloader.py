import argparse
from pytube import YouTube
import os
import os.path as osp
from datetime import date, datetime, timezone, timedelta
import json
import ffmpeg
import glob



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


def download_and_capture(youtube_url, download_dir_path):
    print(f"download: {youtube_url}")
    os.makedirs(download_dir_path, exist_ok=True) # make dir if not exist
    
    ## download mp4 by pytube
    yt = YouTube(youtube_url)
    meta_info = {}
    stream = yt.streams.filter(type="video", file_extension="mp4").order_by("resolution").desc()[0]
    meta_info["filename"] = get_current_day_time()+".mp4"
    meta_info["title"] = stream.title
    meta_info["description"] = yt.description
    meta_info["vcodec"] = stream.video_codec
    meta_info["fps"] = stream.fps
    meta_info["itag"] = stream.itag
    meta_info["type "] = stream.type
    meta_info["resolution"] = stream.resolution
    meta_info["length"] = yt.length
    meta_info["image_root"] = osp.join(download_dir_path, meta_info["filename"].split(".")[0])
    stream.download(output_path = download_dir_path, filename=meta_info["filename"])
    file_path = osp.join(download_dir_path, meta_info["filename"])

    ## [width, height] checked by ffmpeg-python
    probe = ffmpeg.probe(file_path)
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    meta_info["width"] = int(video['width'])
    meta_info["height"] = int(video['height'])
    save_meta_info(meta_info, osp.abspath(file_path)) # call meta info
    print("download path is", file_path)

    ## for capture all vod's frame, we use ffmpeg cli
    img_capture_dir_path = osp.join(download_dir_path, meta_info["filename"].split(".")[0])
    os.makedirs(img_capture_dir_path, exist_ok=True)

    os.system("chmod u+x FILE ./ffmpeg") # 
    os.system("./ffmpeg " + 
    f"-i {file_path} " +
            "-ss 00:00:0 -t 120 " + # if you want slice videos input -t <sec> command
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