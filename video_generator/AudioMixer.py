import ffmpeg
import os.path as osp

# audio mixing time: 5min 35s
def mix_audio_video(video_path:str, meta_info:dict, output_dir:str)-> None:
    file_name = osp.basename(video_path).split('.')[0] + "_mixed_audio.mp4" # final name
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(meta_info['audio_root'])
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(osp.join(output_dir,file_name)).run()