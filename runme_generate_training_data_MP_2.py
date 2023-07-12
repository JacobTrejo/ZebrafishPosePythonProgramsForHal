import time

from scipy.io import savemat
import numpy as np
import os
import cv2 as cv
from FunctionsFile import *
from skimage.util import random_noise
from scipy.io import loadmat
import multiprocessing
import psutil

import pdb
def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def uint8(a):
    a = roundHalfUp(a)

    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

#Got to check if this is also right
def imGaussNoise(image,mean,var):
    # row,col,ch= image.shape
    # sigma = var**0.5
    # gauss = np.random.normal(mean,sigma,(row,col,ch))
    # gauss = gauss.reshape(row,col,ch)
    # noisy = image + gauss

    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


pi = np.pi
#addpath('../')
proj_params = 'proj_params_101019_corrected_new'
proj_params = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/'+proj_params
lut_b_tail = 'lut_b_tail.mat'
lut_s_tail = 'lut_s_tail.mat'
dataset_name = '100_percent'
fish_shapes = 'generated_pose_100_percent.mat'
patchy_noise = 1

# % Load data
lut_s_tail_mat = loadmat(lut_b_tail)
lut_b_tail = lut_s_tail_mat['lut_b_tail']

lut_s_tail_mat = loadmat(lut_s_tail)
lut_s_tail = lut_s_tail_mat['lut_s_tail']

data_dir = ['training_data_3D_', dataset_name]

x_all_data = loadmat(fish_shapes)
x_all_data = x_all_data['generated_pose']
x_all_data = np.array(x_all_data)

idx = 1
imageSizeX = 141
imageSizeY = 141
path = os.path.join("./",data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_pose')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("./",data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_crop_coor')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("./",data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_eye_coor')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("./",data_dir[0] + dataset_name)
path = os.path.join(path,'images')
if not os.path.exists(path):
  os.makedirs(path)

path = os.path.join("./",data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_coor_3d')
if not os.path.exists(path):
  os.makedirs(path)




zeros_mat = np.zeros((141, 141))
startTime = time.time()




def genImage(idx):
    x = np.copy(x_all_data[idx-1,:])
    x[3] = np.random.rand() * 2 * pi
    x[12] = x_all_data[idx-1, 17-1]
    x[0] = (np.random.rand() - 0.5) * 40
    x[1] = (np.random.rand() - 0.5) * 40
    x[2] = (np.random.rand() - 0.5) * 35 + 72.5
    x[3] = np.random.rand() * 2 * pi
    x[4: 12] = x_all_data[idx-1, 0: 8]
    x = x[0:13]
    temp = 99
    temp2 = np.concatenate((x,x_all_data[idx-1, 8: 16],[temp]),axis=0)
    #x[13: 21] = x_all_data[idx-1, 8: 16]
    temp2[21] = x_all_data[idx-1, 18-1]
    x = temp2
    fishlen = np.random.normal(3.8, 0.15)
    seglen = fishlen * 0.09


    x[3] = np.random.rand() * 2 * pi

    [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, coor_3d] = return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY)

    cent_b = np.array([0, 0])
    cent_s1 = np.array([0,0])
    cent_s2 = np.array([0,0])

    #eye_coor = np.array([eye_b, eye_s1, eye_s2])
    eye_coor = np.concatenate((eye_b,eye_s1,eye_s2),axis=1)
    #check if this is the right round function
    filter_size = 2 * roundHalfUp(np.random.rand(3)) + 3
    sigma = np.random.rand(3) + 0.5

##############################################
    #gray_b = imgaussfilt(gray_b, sigma[0], 'FilterSize', filter_size[0])
    kernel = cv.getGaussianKernel(int(filter_size[0]), sigma[0])
    gray_b = cv.filter2D(gray_b,-1,kernel)

    gray_b = uint8(gray_b)

    #gray_s1 = imgaussfilt(gray_s1, sigma[1], 'FilterSize', filter_size[1])
    kernel = cv.getGaussianKernel(int(filter_size[1]), sigma[1])
    gray_s1 = cv.filter2D(gray_s1,-1,kernel)

    gray_s1 = uint8(gray_s1)

    
    kernel = cv.getGaussianKernel(int(filter_size[2]), sigma[2])
    gray_s2 = cv.filter2D(gray_s2,-1,kernel)

    gray_s2 = uint8(gray_s2)




    #Converting to range [0,1]
    gray_b = gray_b / max(gray_b.flatten())
    gray_b = imGaussNoise(gray_b, (np.random.rand() * np.random.normal(50, 10)) / 255, (np.random.rand() * 50 + 20) / 255 ** 2)
    #Converting Back
    gray_b = gray_b * (255/ max(gray_b.flatten()))
    gray_b = uint8(gray_b)



    #Converting to range [0,1]
    gray_s1 = gray_s1 / max(gray_s1.flatten())
    gray_s1 = imGaussNoise(gray_s1, (np.random.rand() * np.random.normal(20, 10)) / 255, (np.random.rand() * 50 + 10) / 255 ** 2)
    #Converting Back
    gray_s1 = gray_s1 * (255/ max(gray_s1.flatten()))
    gray_s1 = uint8(gray_s1)



    #Converting to range [0,1]
    gray_s2 = gray_s2 / max(gray_s2.flatten())
    gray_s2 = imGaussNoise(gray_s2, (np.random.rand() * np.random.normal(20, 10)) / 255, (np.random.rand() * 50 + 10) / 255 ** 2)
    #Converting Back
    gray_s2 = gray_s2* (255/ max(gray_s2.flatten()))
    gray_s2 = uint8(gray_s2)



########################################################################################

    if (patchy_noise):
        pvar = np.random.poisson(0.2)
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                centers = np.random.randint(1, high=142, size=(2))
                
                var_mat = np.copy(zeros_mat)

                ####need to check if -1 correct###
                var_mat[centers[0]-1, centers[1]-1] = 1

                ######################################
                #se = strel('disk', ( (2 * np.random.randint(5, high=21))+1, (2 * np.random.randint(5, high=21))+1 )   )
                #var_mat = imdilate(var_mat, se)
                radi = (2 * np.random.randint(5, high=21))+1
                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radi, radi))
                var_mat = cv.dilate(var_mat,se)

                maxB = max(gray_b.flatten())
                gray_b = gray_b/maxB
                #gray_b = imnoise(gray_b, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)
                gray_b = random_noise(gray_b, mode='localvar', local_vars=(var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)+.00000000000000001)
                gray_b = gray_b * (maxB/max(gray_b.flatten()))
                ###############################################################
        pvar = np.random.poisson(0.2)
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                centers = np.random.randint(1, high=142, size=(2))
                var_mat = np.copy(zeros_mat)
                ######################################
                radi = (2 * np.random.randint(5, high=21))+1
                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radi,radi))
                #var_mat = zeros_mat
                var_mat[centers[0]-1, centers[1]-1] = 1
                var_mat = cv.dilate(var_mat,se)
                #gray_s1 = imnoise(gray_s1, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ^ 2)
                
                maxs1 = max(gray_s1.flatten())
                gray_s1 = gray_s1/maxs1
                gray_s1 = random_noise(gray_s1, mode='localvar', local_vars=(var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ** 2)+.00000000000000001)
                ###############################################################
                gray_s1 = gray_s1 * (maxs1/max(gray_s1.flatten()))

        pvar = np.random.poisson(0.2)
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                centers = np.random.randint(1, high=142, size=(2))
                var_mat = np.copy(zeros_mat)
                #se = strel('disk', np.random.randint(5, high=21))
                radi = (2 * np.random.randint(5, high=21))+1

                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radi, radi))

                #var_mat = zeros_mat
                var_mat[centers[0] - 1, centers[1] - 1] = 1
                #var_mat = imdilate(var_mat, se)
                var_mat = cv.dilate(var_mat, se)

                maxs2 = max(gray_s2.flatten())
                gray_s2 = gray_s2/maxs2
                #gray_s1 = imnoise(gray_s1, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ^ 2)
                gray_s2 = random_noise(gray_s2, mode='localvar',local_vars=(var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ** 2)+.00000000000000001)
                gray_s2 = gray_s2 * (maxs2/max(gray_s2.flatten()))

    gray_b = gray_b * (255 / float(max(gray_b.flatten())))
    gray_s1 = gray_s1 * (255 / float(max(gray_s1.flatten())))
    gray_s2 = gray_s2 * (255 / float(max(gray_s2.flatten())))

    #im = cat(1, gray_b, gray_s1, gray_s2)
    im = np.concatenate((gray_b,gray_s1,gray_s2),axis=0)
    #imwrite(im, [data_dir, '/images/im_', str(idx), '.png'])

    #directory = '/Users/jacob/PycharmProjects/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/images'
    path = 'training_data_3D_100_percent/images'
    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/images'

    os.chdir(path)

    filename = 'im_'+ str(idx) + '.png'
    cv.imwrite(filename,im)



    crop_coor = np.concatenate((crop_b,crop_s1,crop_s2),axis=0)
    #pose = np.array([c_b, c_s1, c_s2])
    pose = np.concatenate((c_b,c_s1,c_s2),axis=1)

    if (np.any(eye_coor)==False or np.any(coor_3d)==False or np.any(pose)==False):
        print(x)
        #print(coor_mf_matname)


    path = 'annotations_' + dataset_name + '_coor_3d'
    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/'+path
    os.chdir(path)
    savemat('coor_3d_'+str(idx)+'.mat', {'coor_3d': coor_3d})

    path = 'annotations_' + dataset_name + '_pose'
    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/'+path
    os.chdir(path)
    savemat('pose_ann'+str(idx)+'.mat', {'pose': pose})


    path = 'annotations_' + dataset_name + '_crop_coor'
    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/'+path
    os.chdir(path)
    savemat('crop_coor_ann_'+str(idx)+'.mat', {'crop_coor': crop_coor})


    path ='annotations_' + dataset_name + '_eye_coor'
    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal/training_data_3D_100_percent/'+path
    os.chdir(path)
    savemat('eye_coor_ann_'+str(idx)+'.mat', {'eye_coor': eye_coor})


    path = '/home/jbtrejo2/TranslatedFishPoseInPython_ForHal'
    os.chdir(path)
    # idx = idx + 1
    if (idx% 1) == 0:
        #print(i)
        print(idx)


import multiprocessing
from multiprocessing import Pool, cpu_count

def init_pool_processes():
    np.random.seed()


if __name__ == '__main__':

    print('Process Starting')
    startTime = time.time()
    
    pool_obj = multiprocessing.Pool(initializer=init_pool_processes)
    pool_obj.map(genImage, range(250000, 500000))
    pool_obj.close()
    endTime = time.time()

    print('Finished Running')
    print('Average Time for 1st: '+str((endTime -startTime)/300))
    
    
    
    
    
    

    
        
    


    
    

    
    
    
    
    





    
    

    
