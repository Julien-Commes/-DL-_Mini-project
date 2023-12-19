import cv2
import os

def slice_video(video_path): 
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*'mp4v')
    
    # Vérifier si la vidéo est ouverte
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        exit()

    # Lire les images et les sauvegarder
    frames=[]
    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Vérifier si la lecture de la vidéo est terminée
        if not ret:
            break

        # Sauvegarder chaque image
        frames.append(frame)
        #image_path = f'frame_{frame_count}.png'
        #cv2.imwrite(image_path, frame)

        # Afficher un message pour montrer la progression
        #print(f"Frame {frame_count} sauvegardée")
        frame_count += 1

    # Libérer la capture vidéo
    cap.release()
    return frames, frame_width, frame_height, fps, fourcc

'''frames, frame_width, frame_height, fps, fourcc = slice_video('video_test.mp4')'''

def write_video(video_path, output, frame_width, frame_height, fps, fourcc):
    # Output setup
    current_path = os.getcwd()
    folder_name = "../results"    
    save_dir = os.path.join(current_path, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Le dossier '{folder_name}' a été créé avec succès dans {current_path}")
    else:
        print(f"Le dossier '{folder_name}' existe déjà dans {current_path}")
    video_writer = cv2.VideoWriter(f'{save_dir}/{video_path}', fourcc, fps, (frame_width, frame_height))
    
    for frame in output:
        video_writer.write(frame)
    
    video_writer.release()

'''write_video('video_test.mp4', frames, frame_width, frame_height, fps, fourcc)'''