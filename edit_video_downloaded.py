import cv2 
from ffpyplayer.player import MediaPlayer
import numpy as np
import os
import random
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import uuid
import download

urls = ["https://www.tiktok.com/@daisybloomss/video/7296909136292220203"]


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1])/ 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def zoom(img, zoom_factor):
      return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def crop_video_duration(input_path, output_path, crop_percentage):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to keep
    num_frames_to_keep = int(total_frames * crop_percentage)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Generate a list of frame indices to keep
    frame_indices_to_keep = random.sample(range(total_frames), num_frames_to_keep)


i = 0
for url in urls:
  download.download_video(url)
  #Sources
  source= 'InputVideos\\Downloaded_from_TikTok.mp4'

  #Instances of librarys
  
  while i < 10:
    cap = cv2.VideoCapture(source)
    frame_count = 0
    #Random Variables
    random_degree = random.uniform(-3.0, 3.0)
    scale_size = random.uniform(0.9, 0.92)
    random_brightness= random.uniform(0.8, 1.3)
    random_contrast = random.uniform(0.01, 0.05)
    random_flip = random.choice([0, 1])
    random_crop = random.randint(20, 25)
    random_pixel_coordinate_x = np.random.randint(0, 400, 10)
    random_pixel_coordinate_y = np.random.randint(0, 400, 10)
    random_sound = random.uniform(0.2, 2.0)


    #Shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #RandomFileName
    name ="OutPutVideos\\TikTok_"+str(uuid.uuid4())+".mp4"
    name_with_sound ="OutPutsWithSound\\TikTok_"+str(uuid.uuid4())+".mp4"

    #writing Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, fps, (width,  height))

    while(cap.isOpened()==True):
        ret, frame = cap.read()

        if ret == True :
    
            frame = rotate_image(frame, random_degree) 
            frame = zoom(frame, scale_size)
            frame =cv2.convertScaleAbs(frame,random_contrast, random_brightness)

            frame_height = frame[0]
            frame_width = frame[1]
            print(width, height)
            if random_flip == 0:
                frame = cv2.flip(frame, 1)
            
            for coordinate in random_pixel_coordinate_x:
              frame[random_pixel_coordinate_x, random_pixel_coordinate_y] = (255, 255, 255)

                
            frame = frame[random_crop:-random_crop, random_crop:-random_crop]
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)  
            out.write(frame)
            # cv2.imshow('Frame', frame)
            frame_count+=1
            
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    video_clip = VideoFileClip(name)
    audio_clip = AudioFileClip(source)

    temp_audio_file = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_file, codec="pcm_s16le")

    audio_clip_modified = audio_clip.volumex(random_sound)
    final_clip = video_clip.set_audio(audio_clip_modified)
    final_clip.write_videofile(name_with_sound, codec="libx264", audio_codec="aac")
    os.remove(name)
    print(frame_count)
    crop_video_duration(name_with_sound, "cropped.mp4", 0.9)
    i+=1