import sys
import os
import json
from PIL import Image, ImageOps
import numpy as np
import cv2

pat_dir = "ano_mura/MURA-v1.1/train/"
train_image_dir = "MURA-v1.1/train/"
patname = os.listdir(pat_dir)
save_image_dir = "train_image/"
save_label_dir = "label_image/"
save_anno = "train_ano.txt"
ano_list = ["Medial Border Radius", "Lateral Border Radius", "Wrist Joint Line", "Radius Shaft Center Line", "Proximal Shaft Intersection", "Distal Shaft Intersection"]
anno_fp = open(save_anno, "w")
label_images = []
for i in patname:
    target = pat_dir+i
    if os.path.isdir(target):
        target_dir = os.listdir(target)
        for j in target_dir:
            image_dir = os.listdir(target + "/" + j)
            for k in image_dir:
                image = target + "/" +  j + "/" + k 
                if not os.path.isdir(image):
                    continue
                for s in os.listdir(image):
                    ano_points = np.zeros(len(ano_list) * 4)

                    slice = image+"/"+s
                    slice_image = train_image_dir + i + "/" + j + "/" + k + "/"+ s[:-4] + "png"               

                    #print(slice_image)                    
                    target_image = Image.open(slice_image)
                    target_image.convert('RGB') 
                    save_name =  j+"_" + k+ "_"+s[:-4] + "jpg"

                    print(target_image.size)
                    with open(slice) as fp:
                        annodata = json.load(fp)                       
                        x_scale = 512/target_image.size[0]
                        y_scale = 512/target_image.size[1]

                        for a in annodata["shapes"]:
                            if a["label"] == "Radius":
                                contours = np.array(a["points"]).astype(np.int32) + 50
                                print(contours.shape)
                                img = np.zeros( (512, 512) ) # create a single channel 200x200 pixel black image                                 
                                contours[:, 0] = contours[:, 0] * x_scale - 58
                                contours[:, 1] = contours[:, 1] * y_scale - 50
                                #contours =contours
                                img = cv2.fillPoly(img, pts = [contours], color=255)
                                #resized = cv2.resize(img, (512, 512))
                                print(save_label_dir + save_name)
                                cv2.imwrite(save_label_dir + save_name, img)                                    
                            
                            for at in range(len(ano_list)):
                                if a["label"] == ano_list[at]:
                                    ano_points[at*4] = int(a["points"][0][0] * x_scale)
                                    ano_points[at*4+1] = int(a["points"][0][1] * y_scale)
                                    ano_points[at*4+2] = int(a["points"][1][0] * x_scale)
                                    ano_points[at*4+3] = int(a["points"][1][1] * y_scale)
                                            
                    print(slice)
                    
                    target_image = target_image.resize((512, 512))
                    
                    
                    #print(save_name)
                    target_image.save(save_image_dir +save_name)
                    

                    anno_fp.write(j+"_" + k+ "_"+s[:-4] + "jpgA" + " ")
                    anno_fp.write("0 0 255 255 ")
                    #anno_fp.write(str(int(np.min(ano_points[0::2]))) + " ")
                    #anno_fp.write(str(int(np.min(ano_points[1::2]))) + " ")
                    #anno_fp.write(str(int(np.max(ano_points[0::2]))) + " ")
                    #anno_fp.write(str(int(np.max(ano_points[1::2]))) + " ")
                    for out in ano_points:
                        anno_fp.write(str(int(out))+" ")
                    anno_fp.write("\n")

anno_fp.close()

                    
            
                
