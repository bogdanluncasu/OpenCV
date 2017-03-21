#import cv2
#import numpy as np
#import matplotlib.pyplot as plt

#print(cv2.__version__)

#img1 = cv2.imread('1.jpg',0)
#img2 = cv2.imread('2.jpg',0)

#def show_rgb_img(img):
#    plt.imshow(cv2.cvtColor(img, cv2.CV_32S))
#    plt.show()

#def to_gray(color_img):
#    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
#    return gray

#img1_gray = img1
#img2_gray = img2

#def gen_sift_features(gray_img):
#    sift = cv2.ORB_create()
#    kp, desc = sift.detectAndCompute(gray_img, None)
#    return kp, desc

#def show_sift_features(gray_img, color_img, kp):
#    plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
#    plt.show()


#kp1, des1 = gen_sift_features(img1_gray)
#kp2, des2 = gen_sift_features(img2_gray)


#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#matches = bf.match(des1,des2)
#matches = sorted(matches,key=lambda x:x.distance)

#cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10],None,flags=True)

#plt.imshow(view)
#plt.show()
#cv2.imwrite("hola.jpg",view)


#import urllib2
#import cv2
#import numpy as np
#import os

#def store_raw_images():
#    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'   
#    neg_image_urls = urllib2.urlopen(neg_images_link).read().decode()
#    pic_num = 1
    
#    if not os.path.exists('neg'):
#        os.makedirs('neg')
        
#    for i in neg_image_urls.split('\n'):
#        try:
#            print(i)
#            imgRequest = urllib2.Request(i)
#            imgData = urllib2.urlopen(imgRequest).read()

#            with open("neg/"+str(pic_num)+".jpg","wb") as f:
#                f.write(imgData)

#            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
#            # should be larger than samples / pos pic (so we can place our image on it)
#            resized_image = cv2.resize(img, (100, 100))
#            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
#            pic_num += 1
            
#        except Exception as e:
#            print(str(e))

#def create_pos_n_neg():
#        for img in os.listdir('neg'):
#                line = 'neg'+'/'+img+'\n'
#                with open('bg.txt','a') as f:
#                    f.write(line)



#watch_cascade = cv2.CascadeClassifier('data/cascade.xml')

#img = cv2.imread('watch.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#watches = watch_cascade.detectMultiScale(gray, 960, 640)
#for (x,y,w,h) in watches:
#       cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

#cv2.imshow('img',img)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()


import numpy as np
import cv2
from cv2 import ml
import matplotlib.pyplot as plt

img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('1.jpg',0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

svm=ml.SVM_create()


matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()