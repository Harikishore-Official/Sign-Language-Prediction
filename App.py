
from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request



import smtplib

#from cloudant.client import  Cloudant



#client = Cloudant.iam("2cbf492c-e69a-4763-8424-f608b7a3259e-bluemix","uDu-_l8TtYZbHHtBwW2Q0NvX_nhEhVAQ0pZdHfpYGpHp",connect=True)
#my_database = client.create_database("database")

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

app.config['DEBUG']


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/VoicetoText")
def VoicetoText():
    return render_template('index1.html')

@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')



@app.route("/Prediction")
def Prediction():
    return render_template('UserHome.html')












@app.route("/start", methods=['GET', 'POST'])
def start():
    error = None
    if request.method == 'POST':
        if request.form["submit"] == "AlphabetDetection":
            import csv
            import copy
            import cv2 as cv
            import mediapipe as mp
            from model import KeyPointClassifier
            from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            args = get_args()
            cap_device = args.device
            cap_width = args.width
            cap_height = args.height

            use_static_image_mode = args.use_static_image_mode
            min_detection_confidence = args.min_detection_confidence
            min_tracking_confidence = args.min_tracking_confidence

            cap = cv.VideoCapture(cap_device)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            keypoint_classifier = KeyPointClassifier()

            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                keypoint_classifier_labels = csv.reader(f)
                keypoint_classifier_labels = [
                    row[0] for row in keypoint_classifier_labels
                ]

            flag = 0

            import win32com.client as wincl
            speak = wincl.Dispatch("SAPI.SpVoice")

            while True:
                key = cv.waitKey(10)
                if key == 27:  # ESC
                    break

                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)
                debug_image = copy.deepcopy(image)
                # print(debug_image.shape)
                # cv.imshow("debug_image",debug_image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # print(hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)

                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                        debug_image = draw_landmarks(debug_image, landmark_list)
                        flag += 1
                        print(flag)
                        if (flag == 100):
                            flag = 0

                            speak.Speak(keypoint_classifier_labels[hand_sign_id])

                        debug_image = draw_info_text(
                            debug_image,
                            handedness,
                            keypoint_classifier_labels[hand_sign_id])

                cv.imshow('Hand Gesture Recognition', debug_image)

            cap.release()
            cv.destroyAllWindows()
        else:

            import cv2
            import numpy as np
            import mediapipe as mp
            import tensorflow as tf
            from tensorflow.keras.models import load_model

            # initialize mediapipe
            mpHands = mp.solutions.hands
            hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            mpDraw = mp.solutions.drawing_utils

            # Load the gesture recognizer model
            model = load_model('mp_hand_gesture')

            # Load class names
            f = open('gesture.names', 'r')
            classNames = f.read().split('\n')
            f.close()
            print(classNames)

            # Initialize the webcam
            cap = cv2.VideoCapture(0)

            while True:
                # Read each frame from the webcam
                _, frame = cap.read()

                x, y, c = frame.shape

                # Flip the frame vertically
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get hand landmark prediction
                result = hands.process(framergb)

                # print(result)

                className = ''

                # post process the result
                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            # print(id, lm)
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)

                            landmarks.append([lmx, lmy])

                        # Drawing landmarks on frames
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                        # Predict gesture
                        prediction = model.predict([landmarks])
                        # print(prediction)
                        classID = np.argmax(prediction)
                        className = classNames[classID]

                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                # Show the final output
                cv2.imshow("Output", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            # release the webcam and destroy all active windows
            cap.release()

            cv2.destroyAllWindows()










    return render_template('UserHome.html')













if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
