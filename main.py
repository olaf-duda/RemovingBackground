import os
import cv2
import numpy as np

# assign directory
directory = 'Potato___Late_blight_seg'
directorymy = 'Potato___Late_blight'

# iterate over files in
# that directory
sum_coef = 0
sum_iou = 0
i = 0

worst_coef = [0, 0, 0, 0, 0]
worst_filename = ['', '', '', '', '']


for filename in os.listdir(directory):
    pathf = os.path.join(directory, filename)
    if os.path.isfile(pathf):

        # ------------ Reading pictures ------------

        # Need to change the path for the proper database

        img_seg = cv2.imread('C:/PUTYOURDIRECTORYHERE/' + filename)
        img = cv2.imread('C:/PUTYOURDIRECTORYHERE/' + filename[:len(filename) - 17] + '.JPG')

        # ------------ Ground truth ------------
        img_gry = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        img_gry_blr = cv2.GaussianBlur(img_gry, (9, 9), 0)

        ret, tsh = cv2.threshold(img_gry_blr, 20,
                                 255, cv2.THRESH_BINARY)

        # ------------ Finding my answer ------------

        hsv = cv2.GaussianBlur(img, (9, 9), 0)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

        # Finding mask of the leaf
        lower_green = np.array([30, 45, 35])
        upper_green = np.array([100, 255, 255])

        mask_g = cv2.inRange(hsv, lower_green, upper_green)

        # Finding mask of the sickness
        lower_brown = np.array([11, 35, 45])
        upper_brown = np.array([35, 255, 240])

        mask_b = cv2.inRange(hsv, lower_brown, upper_brown)

        # Morphology
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE,
                                  kernel=np.ones((3, 3), dtype=np.uint8))

        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE,
                                  kernel=np.ones((3, 3), dtype=np.uint8))

        mask_f = cv2.morphologyEx(mask_b + mask_g, cv2.MORPH_CLOSE,
                                  kernel=np.ones((3, 3), dtype=np.uint8))
        mask_f = cv2.morphologyEx(mask_f, cv2.MORPH_OPEN,
                                  kernel=np.ones((3, 3), dtype=np.uint8))


        # Dice Coefficient
        dice_coef = np.sum(mask_f & tsh)*2 / (np.sum(tsh) + np.sum(mask_f))
        sum_coef += dice_coef
        if dice_coef > worst_coef[0]:
            worst_coef[0] = dice_coef
            worst_filename[0] = filename

            worst_filename = [l for _, l in sorted(zip(worst_coef, worst_filename))]
            worst_coef.sort()

        # IoU
        iou_step = np.sum(mask_f & tsh) / np.sum(mask_f | tsh)
        sum_iou += iou_step

        i += 1

        ## To look at each output
        # masked = cv2.bitwise_and(img, img, mask=mask_f)
        # cv2.imshow('img', masked)
        # cv2.imshow('ori', img_seg)
        # cv2.imshow('theirs', tsh)
        # cv2.imshow('maskl', mask_f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


#Mean of IoU and Dice
print("Dice coefficient:")
print(sum_coef/i)
print("Dice IoU:")
print(sum_iou/i)
