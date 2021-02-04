import os
import cv2
import sys

argument = sys.argv
SHAPE = (1024,1024)

if len(argument) > 1:
    target = argument[1]
else:
    target = "bone"

path = os.path.join(target, "img", "")
file_list = os.listdir(path)
total_count = 0

if not os.path.exists( os.path.join(target, 'crop', "" )):
    os.makedirs(os.path.join(target, 'crop', "" ))
if not os.path.exists(os.path.join(target, 'label_crop', "" )):
    os.makedirs(os.path.join(target, 'label_crop', "" ))
if not os.path.exists(os.path.join(target, 'img', "" )):
    os.makedirs(os.path.join(target, 'img', "" ))
if not os.path.exists(os.path.join(target, 'label_gray', "" )):
    os.makedirs(os.path.join(target, 'label_gray', "" ))

for i in file_list:
    print(total_count)
    #total_count = total_count +1
    #if total_count % 1000 == 0:
    #    print(total_count)
    IMG_DIR = os.path.join(target, "img",  i)
    LABEL_DIR = os.path.join(target, "label_gray",  i)
    print(LABEL_DIR)
    src = cv2.imread(IMG_DIR, cv2.IMREAD_COLOR)
    h, w, c =src.shape
    stride = 128

    dst = src.copy() 
    dist = []
    dist.append(src)
    
    try:
        label_src = cv2.imread(LABEL_DIR, cv2.IMREAD_GRAYSCALE)
        label_prepro = label_src.copy()
        label_prepro[label_prepro > 0] = 255

        label_dist = []
        label_dist.append(label_src)
    except:
        continue

    count = 0
    for j in dist:
        count = count +1
        total_count = total_count +1
        CROP_OUT_DIR = os.path.join(target, "crop",  i[:-4] + "_" + str(count) + ".jpg" )
        cv2.imwrite(CROP_OUT_DIR, j)

        height, width, channel = j.shape        
        j = cv2.flip(j, 1)

        count = count +1
        #total_count = total_count +1
        CROP_OUT_DIR = os.path.join(target, "crop",  i[:-4] + "_" + str(count) + ".jpg" )
        cv2.imwrite(CROP_OUT_DIR, j)

        for k in range(10):
            height, width, channel = j.shape
            matrix = cv2.getRotationMatrix2D((width/2, height/2), k*10, 1)
            dst = cv2.warpAffine(j, matrix, (width, height))
            j=dst

            count = count +1
            #total_count = total_count +1
            CROP_OUT_DIR = os.path.join(target, "crop",  i[:-4] + "_" + str(count) +"angle_" + str(k*10) + ".jpg" )
            cv2.imwrite(CROP_OUT_DIR, j)
                    
            height, width, channel = j.shape        
            j = cv2.flip(dst, 1)

            count = count +1
            #total_count = total_count +1
            CROP_OUT_DIR = os.path.join(target, "crop",  i[:-4] + "_" + str(count) +"angle_" + str(k*10) + ".jpg" )
            cv2.imwrite(CROP_OUT_DIR, j)

    count =0
    for j in label_dist:
        count = count +1
        #total_count = total_count +1
        LABEL_CROP_OUT_DIR = os.path.join(target, "label_crop", i[:-4] + "_" +str(count) +".jpg")
        cv2.imwrite(LABEL_CROP_OUT_DIR, j)

        height, width = j.shape
        j = cv2.flip(j, 1)

        count = count +1
        #total_count = total_count +1
        LABEL_CROP_OUT_DIR = os.path.join(target, "label_crop", i[:-4] + "_" +str(count) +".jpg")
        cv2.imwrite(LABEL_CROP_OUT_DIR, j)

        for k in range(10):
            height, width = j.shape
            matrix = cv2.getRotationMatrix2D((width/2, height/2), k*10, 1)
            dst = cv2.warpAffine(j, matrix, (width, height))
            j=dst

            count = count +1
            #total_count = total_count +1
            LABEL_CROP_OUT_DIR = os.path.join(target, "label_crop", i[:-4] + "_" +str(count) +"angle_" + str(k*10) +".jpg")
            cv2.imwrite(LABEL_CROP_OUT_DIR, j)

            height, width = j.shape
            j = cv2.flip(dst, 1)

            count = count +1
            #total_count = total_count +1
            LABEL_CROP_OUT_DIR = os.path.join(target, "label_crop", i[:-4] + "_" +str(count) +"angle_" + str(k*10) +".jpg")
            cv2.imwrite(LABEL_CROP_OUT_DIR, j)
            

    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()~
