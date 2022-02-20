import dlib
import cv2
import numpy as np
from math import hypot
import pyglet
import time


sound =  pyglet.media.load("arcade.wav",streaming=False)
left_sound = pyglet.media.load("sample.wav",streaming=False)



keyboard = np.zeros((600, 1000, 3), np.uint8)
board = np.zeros((400,600,3),np.uint8)
board[:]=255
keys_set_1 = {0: "Q" , 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}



def draw_letter(letter_index,text,light):
    #Tuşlar

    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400


    width =200
    height =200
    th=3
    if light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 255, 0), th)


    #Yazılar
    font_letter = cv2.FONT_ITALIC
    font_scale = 8
    font_th = 3
    text_size = cv2.getTextSize(text,font_letter,font_scale,font_th)[0]
    width_text ,height_text = text_size[0] , text_size[1]
    text_x = int((width-width_text)/2)+x
    text_y = int((height+height_text)/2)+y
    cv2.putText(keyboard,text,(text_x,text_y),font_letter,font_scale,(0,0,255),font_th)



def orta(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)
def gozkapama_oranti(goz_noktalari,facial_landmark) :
    sol_nokta = facial_landmark.part(goz_noktalari[0]).x, facial_landmark.part(goz_noktalari[0]).y
    sag_nokta = facial_landmark.part(goz_noktalari[3]).x, facial_landmark.part(goz_noktalari[3]).y
    ust_orta = orta(facial_landmark.part(goz_noktalari[1]), facial_landmark.part(goz_noktalari[2]))
    alt_orta = orta(facial_landmark.part(goz_noktalari[5]), facial_landmark.part(goz_noktalari[5]))

    dikey_cizgi_uzunluk = hypot((ust_orta[0] - alt_orta[0]),
                                        (ust_orta[1] - alt_orta[1]))
    yatay_cizgi_uzunluk = hypot((sag_nokta[0] - sol_nokta[0]),
                                        (sag_nokta[1] - sol_nokta[1]))

    oranti = yatay_cizgi_uzunluk /dikey_cizgi_uzunluk
    print(oranti)

    return oranti


font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blinking_frame = 0
fps = 0
letter_index = 0
text = ""
while True:

    _, frame = cap.read()
    keyboard[:] = (0,0,0)
    fps += 1
    active_letter = keys_set_1[letter_index]



    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gri)

    for face in faces:
        landmarks = predictor(gri, face)
        sol_goz_oranti = gozkapama_oranti([36, 37, 38, 39, 40, 41], landmarks)
        sag_goz_oranti = gozkapama_oranti([42, 43, 44, 45, 46, 47], landmarks)

        blink_oranti = (sol_goz_oranti + sag_goz_oranti) / 2
        if blink_oranti > 2.8:
            cv2.putText(frame, "Kapali", (50, 150), font, 6, (255, 0, 0))
            blinking_frame += 1
            fps -=1
            active_letter = keys_set_1[letter_index]

            if blinking_frame == 5:
                text += active_letter
                sound.play()
                time.sleep(1)

        else:

            blinking_frame = 0

        sag_goz_alani = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                  (landmarks.part(37).x, landmarks.part(37).y),
                                  (landmarks.part(38).x, landmarks.part(38).y),
                                  (landmarks.part(39).x, landmarks.part(39).y),
                                  (landmarks.part(40).x, landmarks.part(40).y),
                                  (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        sol_goz_alani = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                  (landmarks.part(43).x, landmarks.part(43).y),
                                  (landmarks.part(44).x, landmarks.part(44).y),
                                  (landmarks.part(45).x, landmarks.part(45).y),
                                  (landmarks.part(46).x, landmarks.part(46).y),
                                  (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        height, widht, _ = frame.shape

        mask = np.zeros((height, widht), np.uint8)

        mask = np.zeros((height, widht), np.uint8)
        cv2.polylines(mask, [sag_goz_alani], True, 255, 2)
        cv2.fillPoly(mask, [sag_goz_alani], 255)
        cv2.polylines(mask, [sol_goz_alani], True, 255, 2)
        cv2.fillPoly(mask, [sol_goz_alani], 255)
        son_sag_goz = cv2.bitwise_and(gri, gri, mask=mask)
        son_sol_goz = cv2.bitwise_and(gri, gri, mask=mask)

        min_x = np.min(sag_goz_alani[:, 0])
        max_x = np.max(sag_goz_alani[:, 0])
        min_y = np.min(sag_goz_alani[:, 1])
        max_y = np.max(sag_goz_alani[:, 1])

        min_x2 = np.min(sol_goz_alani[:, 0])
        max_x2 = np.max(sol_goz_alani[:, 0])
        min_y2 = np.min(sol_goz_alani[:, 1])
        max_y2 = np.max(sol_goz_alani[:, 1])

        gri_sag_goz = son_sag_goz[min_y:max_y, min_x:max_x]
        gri_sol_goz = son_sol_goz[min_y2:max_y2, min_x2:max_x2]
        # gri_sag_goz = cv2.cvtColor(sag_goz,cv2.COLOR_BGR2GRAY)
        _, threshold_sag_goz = cv2.threshold(gri_sag_goz, 70, 255, cv2.THRESH_BINARY)
        height, widht = threshold_sag_goz.shape
        _, threshold_sol_goz = cv2.threshold(gri_sol_goz, 70, 255, cv2.THRESH_BINARY)
        height, widht = threshold_sol_goz.shape
        smallest = (2 ** (1 - 6))
        sag_goz_sol_threshold = threshold_sag_goz[0:height, 0:int(widht / 2)]
        sag_goz_sol_threshold_beyaz = cv2.countNonZero(sag_goz_sol_threshold)

        sag_goz_sag_threshold = threshold_sag_goz[0:height, int(widht / 2):widht]
        sag_goz_sag_threshold_beyaz = cv2.countNonZero(sag_goz_sag_threshold)

        sol_goz_sol_threshold = threshold_sol_goz[0:height, 0:int(widht / 2)]
        sol_goz_sol_threshold_beyaz = cv2.countNonZero(sol_goz_sol_threshold)

        sol_goz_sag_threshold = threshold_sol_goz[0:height, int(widht / 2):widht]
        sol_goz_sag_threshold_beyaz = cv2.countNonZero(sol_goz_sol_threshold)

        sag_goz_ust_threshold = threshold_sag_goz[0:int(height / 2), 0:widht]
        sag_goz_ust_threshold_beyaz = cv2.countNonZero(sag_goz_ust_threshold)

        sag_goz_alt_threshold = threshold_sag_goz[int(height / 2):height, 0:widht]
        sag_goz_alt_threshold_beyaz = cv2.countNonZero(sag_goz_alt_threshold)

        sol_goz_ust_threshold = threshold_sol_goz[0:int(height / 2), 0:widht]
        sol_goz_ust_threshold_beyaz = cv2.countNonZero(sol_goz_ust_threshold)

        sol_goz_alt_threshold = threshold_sol_goz[int(height / 2):height, 0:widht]
        sol_goz_alt_threshold_beyaz = cv2.countNonZero(sol_goz_alt_threshold)

        sag_threshler = sol_goz_sag_threshold_beyaz + sag_goz_sag_threshold_beyaz
        sol_threshler = sol_goz_sol_threshold_beyaz + sag_goz_sol_threshold_beyaz
        alt_threshler = sol_goz_alt_threshold_beyaz + sag_goz_alt_threshold_beyaz
        ust_threshler = sol_goz_ust_threshold_beyaz + sag_goz_ust_threshold_beyaz
        genel_toplam1 = sag_threshler + sol_threshler + alt_threshler + ust_threshler




        if genel_toplam1 > 850:

            if sol_threshler > sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)

            elif sol_threshler < sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sag yukari bakiyorsun"), (50, 200), font, 2, (0, 255, 255), 3)

        if 500 < genel_toplam1 < 850:

            if sol_threshler * 1.1 > sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)

            elif sol_threshler < sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sag yukari bakiyorsun"), (50, 200), font, 2, (0, 255, 255), 3)

        if 300 < genel_toplam1 < 500:

            if sol_threshler * 1.1 > sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)

            elif sol_threshler < sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sag yukari bakiyorsun"), (50, 200), font, 2, (0, 255, 255), 3)

        if genel_toplam1 < 300:

            if sol_threshler * 1.1 > sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)

            elif sol_threshler < sag_threshler and alt_threshler > ust_threshler:
                cv2.putText(frame, str("sag yukari bakiyorsun"), (50, 200), font, 2, (0, 255, 255), 3)
            elif sag_threshler == sol_threshler:

                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)
            elif sag_threshler == sol_threshler:

                cv2.putText(frame, str("sola yukari bakiyorsun"), (50, 100), font, 2, (255, 0, 255), 3)

        if fps == 10:
            letter_index += 1
            fps = 0


        if letter_index == 15 :
            letter_index = 0

        for i in range(15):
            if i == letter_index:
                light = True
            else:
                light = False


            draw_letter(i, keys_set_1[i], light)



            #print(genel_toplam1)
            #print(fps_sag_sol)

    cv2.imshow("Frame", frame)
    cv2.imshow("Klavye",keyboard)
    cv2.putText(board,text,(70,70),font,4,0,3)
    cv2.imshow("board",board)


    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()