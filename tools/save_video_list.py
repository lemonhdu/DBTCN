import os

videos_dir = r"<video path>"
videos_list = os.listdir(videos_dir)
videos_list.sort()
save_txt_dir = "./video_list.txt"

with open(save_txt_dir, "w") as f:
    for item in videos_list:
        video_name = item + "\n"
        f.write(video_name)
