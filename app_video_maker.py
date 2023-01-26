from video_generator.MakingVideo import video_generator

def app_video_maker(df1_name_tagged_GT, meta_info, pred):
    
    video_path_karina = video_generator(df1_name_tagged_GT, meta_info, member='aespa_karina', pred=pred, full_video = True)
    video_path_winter = video_generator(df1_name_tagged_GT, meta_info, member='aespa_winter', pred=pred, full_video = True)
    video_path_ningning = video_generator(df1_name_tagged_GT, meta_info, member='aespa_ningning', pred=pred, full_video = True)
    video_path_giselle = video_generator(df1_name_tagged_GT, meta_info, member='aespa_giselle', pred=pred, full_video = True)
    
    return [video_path_karina, video_path_winter, video_path_ningning, video_path_giselle]

