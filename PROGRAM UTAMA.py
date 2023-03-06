import RPi.GPIO as GPIO
import numpy as np
import joblib
import cv2
import os
import time
from RPLCD import *
from RPLCD.i2c import CharLCD
import matplotlib.pyplot as plt

lcd = CharLCD('PCF8574',0x27)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def run_program():
    start_time = time.time()
    lcd.clear()
    model_load = joblib.load("/home/pi/Desktop/BILL/knn.sav")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    currentframe = 0
    while True:
        success, frame = cam.read()
        cv2.imwrite('/home/pi/Desktop/image' + str(currentframe) + '.jpg', frame)
        currentframe += 1

        if currentframe>1:
            break

    #image = cv2.imread("FILE_TESTING/image1.jpg")
    image = cv2.imread("/home/pi/Desktop/image1.jpg")
    #image = cv2.imread("/home/pi/Desktop/bpw3_outcloudy_35.jpg")
    rotated_image = cv2.rotate(image,cv2.ROTATE_180)
    hsv_img = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
    low_green = np.array([30, 34, 0])
    high_green = np.array([80, 255, 140])
    thres_img = cv2.inRange(hsv_img, low_green, high_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    closed_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('/home/pi/Desktop/segmented.jpg', cv2.bitwise_and(rotated_image,rotated_image, mask=thres_img))

    totalLabels, label, stats, centroid = cv2.connectedComponentsWithStats(closed_img)
    im_result = np.zeros(label.shape)
    addr = []
    count = 0
    for i in range(0, totalLabels):
        if stats[i, 4] < 6000:
            addr.append(i)
        else:
            im_result[label == i] = 1 + count
            count += 1
        addr.sort(reverse=True)
    for i in addr:
        stats = np.delete(stats, i, axis=0)

    start = 1
    stop = stats.shape[0]
    pot_pakai = ['A:TIDAK ADA BAYAM','B:TIDAK ADA BAYAM','C:TIDAK ADA BAYAM','D:TIDAK ADA BAYAM']
    for pot_number in range(start, stop):
        #print("POT KE " + str(pot_number))
        #plt.imshow(im_result == pot_number + 1, cmap='gray')
        #plt.show()
        areacontour = stats[pot_number, cv2.CC_STAT_AREA]
        luasbb = stats[pot_number, cv2.CC_STAT_WIDTH] * stats[pot_number, cv2.CC_STAT_HEIGHT]
        axis_x = stats[pot_number, cv2.CC_STAT_LEFT]
        axis_y = stats[pot_number, cv2.CC_STAT_TOP]
        #print("AREA KONTUR " + str(areacontour))
        #print("LUAS BB " + str(luasbb))
        x_val = np.array([areacontour, luasbb]).reshape(1, -1)
        hasil_klasifikasi = model_load.predict(x_val)
        if hasil_klasifikasi == [1]:
            status = "SIAP PANEN"
        elif hasil_klasifikasi == [0]:
            status = "BELUM PANEN"
        if 0 <= axis_x <= 450 and 0 <= axis_y <= 250:
            pot_pakai[0] = "A:" + status
        elif axis_x > 450 and 0 <= axis_y <= 250:
            pot_pakai[1] = "B:" + status
        elif 0 <= axis_x <= 450 and axis_y > 250:
            pot_pakai[2] = "C:" + status
        elif axis_x > 450 and axis_y > 250:
            pot_pakai[3] = "D:" + status
        pot_pakai.sort()

    #lcd.cursor_pos = (1,1)
    #lcd.write_string(pot_pakai[1])
    for i,pot in enumerate(pot_pakai):
        lcd.cursor_pos = (i,2)
        lcd.write_string(pot)
        #print(pot)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time.sleep(4)

    lcd.clear()
    lcd.cursor_pos = (1,2)
    lcd.write_string("WAKTU KOMPUTASI")
    lcd.cursor_pos = (2,5)
    lcd.write_string("%.2f DETIK" % elapsed_time)
    time.sleep(4)
    lcd.clear()
while True:
    lcd.cursor_pos = (1,2)
    lcd.write_string("TOMBOL START ==>")
    # Wait for the pushbutton to be pressed
    if GPIO.wait_for_edge(13, GPIO.FALLING):
        # Delay for 50 milliseconds to allow the button bounce to settle
        time.sleep(0.10)
        # Check the button state again
        if GPIO.input(13) == GPIO.HIGH:
            # Run the program
            run_program()