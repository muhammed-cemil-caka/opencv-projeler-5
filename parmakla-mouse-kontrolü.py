import cv2
import mediapipe as mp
import pyautogui

# Ekran boyutlarını al
ekran_genislik, ekran_yukseklik = pyautogui.size()

kamera = cv2.VideoCapture(0)

mpEl = mp.solutions.hands
el = mpEl.Hands()
mpCiz = mp.solutions.drawing_utils

while True:
    basarili, kare = kamera.read()
    kare = cv2.flip(kare, 1)
    imgRGB = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    sonuclar = el.process(imgRGB)

    if sonuclar.multi_hand_landmarks:
        for elLandmarks in sonuclar.multi_hand_landmarks:
            x, y = 0, 0
            for id, lm in enumerate(elLandmarks.landmark):
                h, w, c = kare.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  
                    x, y = cx, cy
                    cv2.circle(kare, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4: 
                    tx, ty = cx, cy
                    cv2.circle(kare, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    
            x_ekran = int((x / w) * ekran_genislik)
            y_ekran = int((y / h) * ekran_yukseklik)

            if abs(x - tx) < 20 and abs(y - ty) < 20:
                pyautogui.click(x_ekran, y_ekran)
            else:
                pyautogui.moveTo(x_ekran, y_ekran, duration=0.1)

            mpCiz.draw_landmarks(kare, elLandmarks, mpEl.HAND_CONNECTIONS)

    cv2.imshow("Kare", kare)

    if cv2.waitKey(1) == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
