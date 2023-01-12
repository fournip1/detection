#!/usr/bin/python3
#-- coding: utf-8 --

##################################
# IMPORT DES MODULES NECESSAIRES #
##################################

import numpy as np
import time
import cv2
import os, sys
import random
import logging
from datetime import datetime


##################################
# PARAMETRES                     #
##################################

confidence_level = 0.6 # seuil de confiance pour la detection
delai = 1 # delai entre deux captures
seuil = 5 # duree minimale de presence sur les images qui declenche l'evenement de detection
max_images = 600 # pour debug uniquement...
play_music = True # jouer un morceau si detection?
record_video = True # enregistrer une video si detection?
send_mail = True # envoyer un mail si detection, cette option ecrase la precedente
dest = 'fournip1@yahoo.com'
nsbuffer = 10 # taille du buffer de depart
nebuffer = 10 # taille du buffer de depart

DIR = "/home/pi/Documents/CodePython/Detection/Images/" # repertoire courant du projet
VIDEO = "/Caverne/Captures"  # repertoire ou sont stockees les eventuelles captures
PROTOTXT = DIR + "MobileNetSSD_deploy.prototxt" # parametre du reseau de neurones
MODEL = DIR + "MobileNetSSD_deploy.caffemodel" # parametre du reseau de neurones
CLASSES = ["Fond", "Avion", "Bicyclette", "Oiseau", "Bateau",
           "Bouteille", "Bus", "Voiture", "Chat", "Chaise", "Vache", "Table de repas",
           "Chien", "Cheval", "Motocyclette", "Humain", "Plante en pot", "Mouton",
           "Sofa", "Train", "Écran de télé"] # classes d'objets detectes par le reseau
HUMAINS = "/home/pi/Documents/CodePython/Detection/Humains" # repertoire des musiques pour humains
CHATS = "/home/pi/Documents/CodePython/Detection/Chats" # repertoire des musiques pour chats

os.chdir(DIR)
logging.basicConfig(filename="ssd_detection_full_script.log",filemode='a',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.DEBUG,datefmt='%Y-%m-%d %H:%M:%S')



##################################
# PARAMETRES LECTURE AUDIO       #
##################################

def playfile(str):
        os.system("cvlc "+ str + " vlc://quit")

AEXTENSIONS = [".mp3",".aac","flac",".wav",".ogg",".m3u"]


##################################
# PARAMETRES EMAIL               #
##################################

def sendemail(qui,dest,pj):
        os.system("mutt -s '"+qui+" détecté.e' " + dest + " < mail.txt -a " + pj)

##################################
# LISTER LES MORCEAUX AUDIO      #
##################################

morceauxh = [] # morceaux des humains
for path, dirs, files in os.walk(HUMAINS):
        for filename in files:
                if filename[-4:] in AEXTENSIONS:
                        morceauxh.append(os.path.join(path, filename))


morceauxc = [] # morceaux des chats
for path, dirs, files in os.walk(CHATS):
        for filename in files:
                if filename[-4:] in AEXTENSIONS:
                        morceauxc.append(os.path.join(path, filename))



##################################
# CHARGER LES FLUX ET LE MODELE  #
##################################

os.chdir(DIR)
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL) # reseau de neurone
cap = cv2.VideoCapture(0) # webcam
ret = True
# Verification de la camera
if (cap.isOpened() == False):
	logging.error("Erreur dans l'ouverture de la webcam")
else:
        logging.info("Ouverture de la webcam")


##################################
# LANCER LA CAPTURE              #
##################################


# On entre dans la boucle
trig=0 # le nombre de detections successives
dmusic = False # lancement de la musique
dvideo = False # lancement de la video
# im_count=0
tt=[0,0,0,0,0]
buffer = [] # buffer du debut
qui = ""
while ret and (tt[4]<=8 or tt[4]>=12):
        ret, frame = cap.read()
        otrig=trig # on stocke l'ancienne valeur du compteur pour comparaison
        if record_video or send_mail: # on stocke dans le buffer
                buffer.append(frame)
                if not(dvideo) and len(buffer) > nsbuffer:
                        del buffer[0]
                elif dvideo and len(buffer) > nsbuffer+nebuffer: # on cree la video et on vide le buffer
                        logging.info("Enregistrement d'une capture")
                        now=datetime.now()
                        strvideo=VIDEO + '/capture_'+now.strftime("%Y%m%d_%H%M%S")+'.avi'
                        out = cv2.VideoWriter(strvideo, cv2.VideoWriter_fourcc(*'MJPG'),int(max(1,1/delai)), (640,480))
                        for f in buffer:
                                out.write(f)
                        out.release()
                        dvideo,trig,otrig = False,0,0 # reinitialisation des triggers
                        buffer = []
                        if send_mail:
                                logging.info("Envoi de l'email")
                                sendemail(qui,dest,strvideo)
                        qui = ""


        R = np.asarray(frame[:,:,0])
        G = np.asarray(frame[:,:,1])
        B = np.asarray(frame[:,:,1])
        L=0.2126*R.mean() + 0.7152*G.mean() + 0.0722*B.mean() # calcul de la luminosite
        # im_count+=1
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        t=time.time()
        tt=time.gmtime(t) # pour arreter la machine entre les minutes 8 et 11 (Nounous cam), voir la condition dans le while
        for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                if confidence > confidence_level and L>80:
                        trig+=1 # le nombre de detections successives augmente de 1
                        qui = CLASSES[idx]
                        logging.debug( qui + " detecté!")
                        logging.debug("Niveau de confiance: " + str(round(confidence*100,2)))
                        if qui=="Humain" and dmusic and play_music:
                                playfile(random.choice(morceauxh))
                                dmusic,trig,otrig = False,0,0 # reinitialisation des triggers
                        elif qui=="Chat" and dmusic and play_music:
                                playfile(random.choice(morceauxc))
                                dmusic,trig,otrig = False,0,0 # reinitialisation des triggers
        
        if trig>otrig: # on est dans un cas de detections successives
                if trig>=seuil-1: # on a depasse le seuil
                        dmusic,dvideo = True,True # les triggers de musique et video sont actives
        else: # on remet les compteurs a 0
                trig,otrig = 0,0 # reinitialisation des triggers

        time.sleep(delai)


# Lorsque tout est fini, on libere les flux 
cap.release()

# On ferme toutes les fenetres
cv2.destroyAllWindows()

logging.info("Detection terminée!")
