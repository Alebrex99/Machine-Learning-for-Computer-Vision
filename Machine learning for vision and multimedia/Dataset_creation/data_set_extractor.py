import random
import cv2
import matplotlib.pyplot as plt
import os
import re
import json


def main():
    # init the camera
    video_dir= r'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_trailers'
    output_dir= r'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_frames'
    output_dir_test = r'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_frames_test'

    videos ={
        'A SIlent Voice Official Trailer Netflix.mp4':'A_Silent_Voice',
        'A Silent Voice - Official Trailer-(1080p).mp4': 'A_Silent_Voice',
        'A SILENT VOICE - KOE NO KATACHI 「 AMV 」 Lovely _ Billie Eilish.mp4': 'A_Silent_Voice',
        'Attack on Titan Season 2 Trailer.mp4': 'Attack_on_Titans',
        'Attack on Titan Season 4 Trailer Subbed (HD)-(1080p60).mp4': 'Attack_on_Titans',
        'Attack on Titan Season 3 Part 2 Trailer - English Dubbed-(1080p).mp4': 'Attack_on_Titans',
        'Attack on Titan AMV- Warriors .mp4': 'Attack_on_Titans',
        'Demon slayer season 3 Final trailer 4k ｜ Official with Release date ｜.mp4': 'Demon_Slayer',
        'Demon Slayer - Opening 1 _ 4K _ 60FPS _ Creditless _-(1080p60).mp4': 'Demon_Slayer',
        'Demon Slayer- Warriors.mp4': 'Demon_Slayer',
        'Demon Slayer AMV- BLOOD WATER.mp4': 'Demon_Slayer',
        'Demon Slayer - Shameless [AMV]-(1080p).mp4': 'Demon_Slayer',
        'Demon Slayer_ Rengoku vs Akaza Anime Battle _ Mugen Train _ 4K 60FPS-(1080p60).mp4': 'Demon_Slayer',
        'Jujutsu Kaisen「AMV」- beggin-(1080p).mp4':  'Jujutsu_Kaisen',
        'Jujutsu Kaisen「AMV」- Whatever It Takes-(1080p).mp4': 'Jujutsu_Kaisen',
        'Jujutsu Kaisen  - All Openings (1-3) - 4K _ Creditless-(1080p).mp4': 'Jujutsu_Kaisen',
        'Fullmetal Alchemist Brotherhood - Whatever It Takes - AMV -.mp4':'Full_Metal_Alchemist_Brotherhood',
        'Fullmetal Alchemist Brotherhood (Trailer) (1).mp4': 'Full_Metal_Alchemist_Brotherhood',
        "Fullmetal Alchemist_ Brotherhood [AMV] -  Beggin'-(1080p).mp4": 'Full_Metal_Alchemist_Brotherhood',
        'One piece1(1080p).mp4' : 'One_piece',
        "One Piece AMV - Can't Hold Us-(1080p).mp4": 'One_piece',
        'Death Note _ OFFICIAL TRAILER-(1080p).mp4':'Death_Note',
        'Death Note - English - Fanmade Trailer.mp4': 'Death_Note',
        'Death Note AMV - Rotten Inside.mp4': 'Death_Note',
        'Death Note_Hit and Run-(1080p50).mp4': 'Death_Note',
        'Dragon Ball Super「AMV」Impossible-(1080p60).mp4':'Dragon_Ball',
        'Dragon Ball Z Trailer-(1080p).mp4':'Dragon_Ball',
        'Dragon ball Z_Super AMV - Stay This Way-(1080p).mp4': 'Dragon_Ball',
        'Fruits Basket (2019) English Dub Trailer(720p).mp4':'Fruits_Basket',
        'Fruits Basket (Anime Boston 2020 Best Trailer_Commercial)-(1080p).mp4': 'Fruits_Basket',
        'Fruits Basket-Someone you loved [AMV] - Fruits Basket-(1080p).mp4': 'Fruits_Basket',
        'GINTAMA_ THE FINAL (2021) Trailer _ Anime Feature Film-(1080p).mp4':'Gintama',
        'GINTAMA.mp4': 'Gintama',
        'Hunter X Hunter Set 1- Official Extended Trailer-(1080p).mp4': 'Hunter_X_Hunter',
        'Hunter x Hunter AMV - I Will Show You.mp4': 'Hunter_X_Hunter',
        'Naruto AMV - Catch Fire-(1080p).mp4': 'Naruto',
        "Naruto's life in one minute  - Emotional AMV-(1080p).mp4": "Naruto",
        'My Hero Academia - Official Fan Made Trailer-(1080p).mp4': 'My_Hero_Academia',
        'My Hero Academia - Season One - English Trailer [HD](720p).mp4': 'My_Hero_Academia',
        'My Hero Academia Amv - Remember The Name-(1080p60).mp4': 'My_Hero_Academia',
        'One Punch Man - Trailer HD(720p).mp4': 'One_Punch_Man',
        'One Punch Man AMV - Indestructible(720p).mp4': 'One_Punch_Man',
        'Re_ZERO - Official Trailer-(1080p).mp4': 'Re_ZERO_2',
        'Re_Zero「AMV」- Natural-(1080p).mp4': 'Re_ZERO_2',
        'Steins; Gate_ An Overly Cinematic Trailer-(1080p).mp4': 'Steins;Gate',
        'Steins;Gate - Fan Trailer - English Dub-(1080p).mp4': 'Steins;Gate',
        'Steins;Gate AMV - E.T-(1080p).mp4': 'Steins;Gate',
    }
    #settare il numero di video che si vogliono : MAX = len(videos)
    videoNumber = len(videos)
    range_frame = 10

    videos = dict(list(videos.items())[42:44]) #video da elaborare, tutti o alcuni (0 - 46)
    labels = []
    frames_dict = {}
    '''
    {"video1" : 
        ["label1", "label", ...]
    "video2" :
        [ ... ]
    }
    '''
    count_frames = 0
    existing_labels = set()
    for video_file, label_name in videos.items():
        # Create a directory with the label name to store the frames for this video
        if label_name in existing_labels:
            frames_dir = os.path.join(output_dir, label_name)
            frames_dir_test = os.path.join(output_dir_test, label_name)
        else:
            # Create a new directory with the label name
            count_frames=0
            frames_dir = os.path.join(output_dir, label_name)
            frames_dir_test = os.path.join(output_dir_test, label_name)

            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(frames_dir_test, exist_ok=True)
            existing_labels.add(label_name)

        '''if label_name in videos.values():
            count_labels = count_labels +1
            frames_dir = os.path.join(output_dir, label_name)'''
        # Open the video file
        video_path = os.path.join(video_dir, video_file) #'C:\Users\Utente\PycharmProjects\machine_learning_DataSet\_trailers\video_i_esimo.mp4'
        video = cv2.VideoCapture(video_path)
        print(video_file)

        # Get the total number of frames in the video
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


        #elimina il _numero alla fine per costruire delle etichette uguali per ogni tipo di video
        label = re.sub(r'_\d+$' , '', label_name)
        frames_dict[video_file] = video_file
        frames_dict[video_file] = []
        # Loop through each frame in the video
        frames_to_extract = range(0, num_frames, range_frame) #frames : 1 ogni 20 : es) video 2 min , 25 fps -> 25*2*60 = 3000/20 = 150

        #TEST FRAMES:
        num_frames_test = int(len(frames_to_extract)*0.1)
        frames_to_extract_test = random.sample(frames_to_extract, num_frames_test )

        for frame_num in frames_to_extract:
            # Set the current frame position to the current frame number
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Read the current frame
            ret, frame = video.read()

            # If the frame was read successfully, save it as an image file
            if ret:
                count_frames=count_frames+1
                #resize de images
                frame = cv2.resize(frame, dsize=(512, 512))
                if frame_num in frames_to_extract_test:
                    frame_path = os.path.join(frames_dir_test, f"{label_name}_{count_frames}.jpg")
                else:
                    frame_path = os.path.join(frames_dir, f"{label_name}_{count_frames}.jpg")
                cv2.imwrite(frame_path, frame)
                labels.append(label)
                frames_dict[video_file].append(label)
        # Release the video file
        video.release()


    json_string = json.dumps(frames_dict) #create json string
    #open() : secondo param =  C:\Users\Utente\PycharmProjects\machine_learning_DataSet
    with open("labels.json", "w") as f:
        f.write(json_string)

    print("Done!")
    print(labels)
    print(frames_dict)



if __name__ == "__main__": #se lancia un eccezione esci ed interrompi flusso video
    try:
        main()
    except KeyboardInterrupt: #interruzione flusso video tramite tastiera
        exit(0)
