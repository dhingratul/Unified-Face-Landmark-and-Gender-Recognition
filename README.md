# Face Detection-Landmarks-Gender Recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Face Detection, landmark detection and gender classification in a unified workflow

# Requirements
1. keras
2. dlib
3. imutils
4. cv2
5. numpy
6. sklearn
7. scipy
8. pandas
9. pickle

# Run
1. Unzip pre-trained weights into model/ from [here](https://drive.google.com/file/d/0B23AMrU3NLB3TEhrWkd0ZW9INVU/view?usp=sharing)
2. cd into src/ directory
2. `python deploy.py` with the default settings
3. `python deploy.py --model 'DL' ` with Deep-learning based  model for gender classification
4. `python deploy.py --model 'DL' --image 'Path to image'` , sample images are present in '../data/'

# Note:
1. Face landmarking is done using dlib
2. Gender Classifier based on Deep Learning, and PCA are both trained on LFW deep funelled dataset
3. Images in ../data/ with prefix '0_{}.jpg' are from LFW and '1_{}.jpg' from COCO dataset
4. Deep Learning based method is two times(2x) slower than simpler model just using PCA, but Deep Learning based method for gender classification is more accurate. See performance of both the models on ../data/0_2.jpg for the image from same distribution(LFW), and ../data/1_1.jpg for image from a different distribution(COCO)
5. The model works well with multiple images in a frame, see ../data/0_6.jpg

# Credits
[Adrian Rosebrock](https://github.com/jrosebr1/imutils) for imutils
