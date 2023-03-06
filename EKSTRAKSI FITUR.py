import cv2
import numpy as np
import glob2 as glob
import pandas as pd

def feature_extraction(number,original_img):
    rotated_image = cv2.rotate(original_img, cv2.ROTATE_180)
    hsv_img = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
    low_green = np.array([30, 34, 0])
    high_green = np.array([80, 255, 160])
    thres_img = cv2.inRange(hsv_img, low_green, high_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    closed_img = cv2.morphologyEx(thres_img, cv2.MORPH_CLOSE, kernel)

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
    all_area_contour = []
    all_luas_bb = []
    for pot_number in range(start, stop):
        areacontour = stats[pot_number, cv2.CC_STAT_AREA]

        all_area_contour.append(areacontour)
        luasbb = stats[pot_number, cv2.CC_STAT_WIDTH] * stats[pot_number, cv2.CC_STAT_HEIGHT]
        all_luas_bb.append(luasbb)
    return all_area_contour, all_luas_bb, pot_number

label = []
area_contour = []
luas_boundingbox = []
classes = ["BELUM PANEN","PANEN"]
img_number = 1
for index,string in enumerate(classes):
    image = glob.glob(f"DATA_FIX/{string}/*.jpg")
    for img in image:
        image_raw = cv2.imread(img)
        feature_areac, feature_lbb, pot_number = feature_extraction(img_number,image_raw)
        for a in feature_areac:
            area_contour.append(a)
        for b in feature_lbb:
            luas_boundingbox.append(b)
        for i in range(pot_number):
            if string == classes[0]:
                label.append(0)
            else:
                label.append(1)
        img_number+=1
df = pd.DataFrame({"Label": label, "Area_Contour": area_contour, "Luas_Bounding_Box": luas_boundingbox})
df.to_csv("DATA_BAYAM.csv", index=False)