import cv2

def slice_video(video_path): 
    # Charger la vidéo
    video_path = video_path
    cap = cv2.VideoCapture(video_path)

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
    return frames

slice_video('video_test.mp4')