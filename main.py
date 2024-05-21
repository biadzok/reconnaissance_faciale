import face_recognition
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk

# Charger les images des personnes
image_person1 = face_recognition.load_image_file("img/Amadou.JPG")
image_person2 = face_recognition.load_image_file("img/Marin.JPG")
image_person3 = face_recognition.load_image_file("img/Tristan.JPG")

# Encoder les visages
encoding_person1 = face_recognition.face_encodings(image_person1)[0]
encoding_person2 = face_recognition.face_encodings(image_person2)[0]
encoding_person3 = face_recognition.face_encodings(image_person3)[0]

# Créer des listes de noms et d'encodages
known_face_encodings = [
    encoding_person1,
    encoding_person2,
    encoding_person3
]

# Liste des têtes déjà reconnues
known_face_names = [
    "Amadou",
    "Marin",
    "Tristan"
]

# Démarrer le flux vidéo
video_capture = cv2.VideoCapture(0)

# Boucle principale
def update_frame():
    # lecture de l'image, exit si problème
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image from camera. Exiting.")
        return
    
    # Conversion de l'image pour face-recognition
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # boucle de traitement des visages détectés
    recognized_names = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # calcul des distances des visages et determination du meilleur match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Ajout de la personne à la liste des noms reconnus
        if name != "Unknown":
            recognized_names.append(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mise à jour des labels d'état et des personnes reconnues
    for i, (name, label) in enumerate(zip(known_face_names, person_labels)):
        if name in recognized_names:
            label.config(bg="green")
        else:
            label.config(bg="red")

    # Conversion de l'image pour Tkinter
    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# Initialisation de Tkinter
root = tk.Tk()
root.title("Face Recognition System")

# Ajout d'un cadre pour la fenêtre
frame = Frame(root)
frame.pack(side=tk.LEFT, padx=10, pady=10)

# ajout des labels pour les personnes reconnues 
person_labels = []
for name, image_path in zip(known_face_names, ["img/Amadou.JPG", "img/Marin.JPG", "img/Tristan.JPG"]):
    img = Image.open(image_path)
    img = img.resize((100, 100), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(img)
    panel = Label(frame, image=imgtk)
    panel.image = imgtk
    panel.pack(side=tk.TOP, pady=5)
    label = Label(frame, text=name, font=('Helvetica', 12), bg="red")
    label.pack(side=tk.TOP, pady=5)
    person_labels.append(label)

# création des labels pour afficher la vidéo
video_label = Label(root)
video_label.pack(side=tk.RIGHT, padx=10, pady=10)

# lancement de la fonction de la mise à jour des frames & lancement de tkinter
root.after(10, update_frame)
root.mainloop()

# fermeture de la caméra et libération des ressources
video_capture.release()
cv2.destroyAllWindows()