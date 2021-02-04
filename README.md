# Process method 

1. raw_data_processing

2. dataset augmentation

3. segmentation train and test



## raw_data_processing

1. GET MURA dataset from https://stanfordmlgroup.github.io/competitions/mura/ to raw_data_processing/MURA-v1.1/

2. LOCATE annotation of json file to raw_data_processing/ano_mura/
  - target json annotation consist of vertexs lists

3. RUN raw_data_processing/preprocessing.py

## dataset augmentation

1. COPY raw_data_processing/train_image to dataset/bone/img

2. COPY raw_data_processing/label_image to dataset/bone/label_gray

3. RUN dataset/second_preprocess.py

## segmentation train and test

1. RUN train_dlinknet.py

2. After trainning, RUN test_dlinknet.py




# Brief summary of methods
1. target vertexs of bones segmenation annotations -> target area segmentation annotation

2. Augmentation
- FILP
- ROTATION, each 10 degrees from 0 to 90 degrees

3. Train the data with Dlinknet 
https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html

4. Post processing with laplacian of gaussian to smooth results

5. Get target border line and central line based on Segmentation Results




