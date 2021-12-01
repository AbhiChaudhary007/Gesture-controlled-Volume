import cv2
import mediapipe as mp

def handtracking(img):
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    lmList = []

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hlms, mpHands.HAND_CONNECTIONS)
            for id,cor in enumerate(hlms.landmark):
                # print(id, cor)
                h, w, c = img.shape
                cx, cy = int(cor.x*w), int(cor.y*h)
                lmList.append([id, cx, cy])
                print(lmList)
                return (id, cx, cy)


if __name__ == '__main__':
    while True:
        cap = cv2.VideoCapture(0)
        _, img = cap.read()
        print(handtracking(img))

        cv2.imshow('Image', cv2.flip(img,1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
