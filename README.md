# Deploy-an-AI-service-on-Microsoft-Azure-cloud

import numpy as np
import cv2
import time
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
import cognitive_face as CF
import datetime
from datetime import date
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType



KEY = '41586e4c845a4d6b85fa79c10b2d769b'
ENDPOINT = 'https://faceresourcegroup.cognitiveservices.azure.com/'
# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
PERSON_GROUP_ID = 'my-person-group5'
# Used for the Snapshot and Delete Person Group examples.
# assign a random ID (or name it anything)
TARGET_PERSON_GROUP_ID = str(uuid.uuid4())
'''
Create the PersonGroup
'''
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)


# Define woman friend
woman = face_client.person_group_person.create(PERSON_GROUP_ID, "Woman")
# Define man friend
man = face_client.person_group_person.create(PERSON_GROUP_ID, "Man")
# Define child friend
child = face_client.person_group_person.create(PERSON_GROUP_ID, "Child")
# Define client friend
amer = face_client.person_group_person.create(PERSON_GROUP_ID, "amer")
'''
Detect faces and register to correct person
'''
# Find all jpeg images of friends in working directory
woman_images = [file for file in glob.glob('*.jpg') if file.startswith("woman")]
man_images = [file for file in glob.glob('*.jpg') if file.startswith("man")]
child_images = [file for file in glob.glob('*.jpg') if file.startswith("child")]
amer_images = [file for file in glob.glob('*.jpg') if file.startswith("amer")]


# Add to a woman person
for image in woman_images:
    w = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, woman.person_id, w)


# # Add to a man person
for image in man_images:
    m = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, man.person_id, m)


# # Add to a child person
for image in child_images:
    ch = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, child.person_id, ch)

# Add to a amer person
for image in amer_images:
    a = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, amer.person_id, a)



# Training person group
print()
print('Training the person group...')
# Train the person group
face_client.person_group.train(PERSON_GROUP_ID)

while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    print()
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        sys.exit('Training the person group has failed.')
    time.sleep(5)

camera = cv2.VideoCapture(0)
currentframe = 0
camera_photo = ''
while True:
    return_value, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', gray)
    if cv2.waitKey(1) & 0xFF == ord('x'):

        currentframe += 1
        cv2.imwrite('abod' + str(currentframe) + '.jpg', image)
        camera_photo = ('abod' + str(currentframe) + '.jpg')

        break
camera.release()
cv2.destroyAllWindows()
img = open(camera_photo, 'r+b')
img_id = []
img_id = face_client.face.detect_with_stream(img)
for img in img_id:
    print(img.face_id)


currentDT = datetime.datetime.now()
found = 0
for person in face_client.person_group_person.list(PERSON_GROUP_ID):
    found = 0
    for faceID in img_id:
        verf_face = face_client.face.verify_face_to_person(faceID.face_id, person.person_id, PERSON_GROUP_ID)
        if(verf_face.confidence > .5): 
            found = 1
            print('face belongs to : ', person.name , '\nwith confidence : ', verf_face.confidence ,'\ntime: ', currentDT.strftime("%c"))
            break
if(found == 0):
    print('Face is not verified')
