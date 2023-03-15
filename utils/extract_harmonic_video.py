import os
import os.path as osp
import cv2
import json


def generate_hr():
    video_dir = './datasets/harmonic'
    output_dir = './datasets/harmonic/original'
    metadata = 'meta_info_harmonic_GT.txt'
    harmonic_videos = ['1 VENICE 5994 UHD h264.mp4', 
                       '2 RALLY 5994 UHD h264.mp4', 
                       '3 fjords 50p uhd 2 h264.mp4', 
                       '4 india buildings 50p uhd 2 h264.mp4',  
                       '6 HongKong_UHD_5994P h264.mp4', 
                       '7 SNOW MONKEYS 5994 UHD h264.mp4', 
                       '8 AMERICAN FOOTBALL 5994 UHD h264.mp4', 
                       '9 Streets_of_India_UHD_50P h264.mp4', 
                       '12 RedRock_Vol3_UHD_50p h264.mp4', 
                        '13 RedRock_Vol2_5994 uhd h264.mp4', 
                        '17 MYANMAR 5994 UHD h264.mp4']

    v = 0
    print('Extracting harmonic videos ...')
    meta_info = []
    for videofn in harmonic_videos:
        if not osp.exists(f'{output_dir}/{v+1:03d}'):
            os.makedirs(f'{output_dir}/{v+1:03d}')

        video = cv2.VideoCapture(f'{video_dir}/{videofn}')
        fps = 120
        i = 0
        while video.isOpened():
            frame_id = int(fps*i)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            if not ret:
                break
            h, w, c = frame.shape
            cv2.imwrite(f'{output_dir}/{v+1:03d}/{i+1:08d}.png', frame)
            i+=1
        
        meta_info.append([v+1, i, h, w, c])
        video.release()
        v += 1

    with open(metadata, 'w') as meta_file:
        meta_file.write(f'{v+1:03d} {i+1} ({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})\n')
            

    meta_file.close()


def generate_lr_BDx4():
    video_dir = './datasets/harmonic/hr'
    output_dir = './datasets/harmonic/lr/BDx4'
    print('Processing lr images ...')
    for root, dirs, files in os.walk(video_dir):
        if not dirs:
            output_lr = f"{output_dir}/{root.split('/')[-1]}"
            if not osp.exists(output_lr):
                os.makedirs(output_lr)
        for f in files:
            image = cv2.imread(f'{root}/{f}')
            h_hr, w_hr = image.shape[0], image.shape[1]
            h_lr, w_lr = h_hr//4, w_hr//4
            image_lr = cv2.resize(image, (h_lr, w_lr), interpolation=cv2.INTER_CUBIC)
            print(f"{f}  hr_shape:{image.shape} lr_shape:{image_lr.shape}")
            cv2.imwrite(f'{output_lr}/{f}', image_lr)


if __name__=='__main__':
    generate_hr()
