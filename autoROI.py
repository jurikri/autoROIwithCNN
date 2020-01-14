# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:06:58 2020

@author: msbak
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import pandas as pd
import random
import pickle

FPS = 3.4
def smoothListGaussian(array1,window):  
     window = round(window)
     degree = (window+1)/2
     weight=np.array([1.0]*window)  
     weightGauss=[]  

     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  

     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(array1.shape[0]-window)
     
     weight = weight / np.sum(weight) # nml

     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(array1[i:i+window])*weight)/sum(weight)  

     return smoothed 

# In[] tif, csv file list
path1 = 'D:\\autoROI_CNN\\'
file_list1 = os.listdir(path1)
dev = True # 개발용 시각화
pathsave=[]; [pathsave.append([]) for u in range(len(file_list1))]
for i1, SE in enumerate(file_list1): 
    file_list2 = os.listdir(path1 + SE)
    pathsave; [pathsave[i1].append([]) for u in range(2)]
    
    for k in range(len(file_list2)):
        file_name = file_list2[k]
        extenstion = os.path.splitext(file_name)[-1] # extension
        full_path = path1 + SE + '\\' + file_name

        if extenstion == '.tif':
            pathsave[i1][0].append(full_path) # 0 for tif

        elif extenstion == '.csv':
            pathsave[i1][1].append(full_path) # 1 for csv
                
# In[] tif, turboreg 오류해결 (이미지를 따로 저장하지 않으니, 다음 세션과 연결할것)
seed = 0; SE= 0
X=[];Y=[];Z=[]
# SE,se session 시작

roi_sequence=[] # 시간 label용 기록
nonroi_sequence=[] # 시간 label용 기록

mshist = []
for SE in range(len(file_list1)):
    print('start at', SE)
    im = io.imread(pathsave[SE][0][0])
    
    # In tif, turboreg 오류해결 (이미지를 따로 저장하지 않으니, 다음 세션과 연결할것)
    c1 = np.isnan(np.mean(np.mean(im, axis=2), axis=1))
    c2 = np.min(np.min(im, axis=2), axis=1) < -100000
    c3 = np.max(np.max(im, axis=2), axis=1) > +100000
    
    diff = np.zeros(im.shape[0]); diff[:] = np.nan
    for frame in range(im.shape[0]):
        diff_inframe = []
        for irow in range(im.shape[1]-1):
            diff_inframe.append(np.mean(np.abs(im[frame,irow,:]-im[frame,irow+1,:])))
        diff[frame] = np.max(diff_inframe)
    c4 = diff > 10000

    error = c1 + c2 + c3 + c4
    
    for i in np.where(error > 0)[0]:
        if dev or True:
            plt.figure()
            plt.imshow(im[i,:,:], cmap='gray')
            plt.title(str(SE) + '_' + str(i))
            im[i,:,:] = im[i-1,:,:]; # 오류 프레임을 이전 프레임으로 대체 
     
    #왜 인지는 모르겠으나 음수값이 존재함. 모두 0으로 대체
    im[im<0] = 0

    meanframe = np.mean(im, axis=0) # 데이터로 쓰진않음. ROI 위치 확인용
    roilist = pathsave[SE][1]
    
    rowmax = meanframe.shape[0]
    colmax = meanframe.shape[1]
    
    marker = 1000
    if dev:
        plt.figure()
        plt.title(str(SE)+'_mean_image')
        plt.imshow(meanframe)
        
    # In ROI.csv 순서대로 import,
    tmplabel = np.zeros((rowmax, colmax, len(roilist)))
    for i in range(len(roilist)):
        coordinate = np.array(pd.read_csv(roilist[i], header=None))
        
        col = np.array(np.round(coordinate[:,0]), dtype=int)
        row = np.array(np.round(coordinate[:,1]), dtype=int)
        
        col[np.where(col >= colmax)[0]] = colmax-1
        row[np.where(row >= rowmax)[0]] = rowmax-1
        
        col[np.where(col < 0)[0]] = 0
        row[np.where(row < 0)[0]] = 0

        top = np.min(row); bottom = np.max(row)
        left = np.min(col); right = np.max(col)
        # 경계면을 따라 속을 채움
        for irow in range(top, bottom+1):
            if irow in row:
                col_in_row = col[np.where(row==irow)[0]]
                tmplabel[irow, np.min(col_in_row):np.max(col_in_row)+1, i] = marker
        for icol in range(left, right+1):
            row_in_col = np.where(tmplabel[:, icol, i] == marker)[0]
            tmplabel[np.min(row_in_col):np.max(row_in_col)+1, icol , i] = marker
    
    if dev:
        plt.figure()
        plt.title(str(SE)+'_ROI')
        plt.imshow(meanframe + np.sum(tmplabel, axis=2))
        
    roiframe = np.sum(tmplabel, axis=2)>0
        
    # In[]
    ## dev
    cnn_step_size = 1
    cnn_window_size = 3
    ws = cnn_window_size
    
    # 좌표 설정 (roi, nonroi)
    matrix = np.array(im[0,:,:])
    
    roilist=[];
    for k in range(2):
        roilist.append(np.where(roiframe==1)[k])
    roilist = np.transpose(np.array(roilist))
    
    index_save=[]; label=[]
    for k in range(roilist.shape[0]):
        row = roilist[k][0]
        col = roilist[k][1]
        
        if row-ws >= 0 and col-ws >= 0 and row+ws <= matrix.shape[0] and col+ws <= matrix.shape[1]:
            index_save.append([row,col])
            label.append(1)

    roiNum = roilist.shape[0]
    print(SE, 'roi 면적', (matrix.shape[0] * matrix.shape[1]) > roiNum)
    
    ## non roi 랜덤하게 설정
    possible=[]
    for row in range(ws,matrix.shape[0]-ws):
        for col in range(ws,matrix.shape[1]-ws):
            possible.append([row,col])
    
    random.shuffle(possible)
    roilist2 = []
    for u in range(roilist.shape[0]):
        roilist2.append(list(roilist[u,:]))
    
    cnt = 0; test=np.zeros(matrix.shape)
    for i in range(len(possible)):
        if cnt > roilist.shape[0]:
            break
#            print(cnt)
        row = possible[i][0]
        col = possible[i][1]

        passsw=True
        if [row,col] in roilist2:
            passsw=False

        if passsw:
            index_save.append([row,col])
            label.append(0)
            
            cnt += 1
            test[row,col] = 1
    
    if dev:
        plt.title(str(SE) + ' roi, nonroi position')
        plt.imshow(test + roiframe*2)
    # In[]
    
    rnn_step_size = int(round(FPS*1))
    rnn_window_size = int(round(FPS*6/2))
    rws = rnn_window_size
    
    for i in range(len(index_save)):
        row = index_save[i][0]
        col = index_save[i][1]
        
        mssignal = np.array(im[:,row,col])
        
        raising=[]
        for j in range(mssignal.shape[0]-1):
            raising.append(mssignal[j+1] - mssignal[j])
        
        elit = np.argsort(raising)[::-1][:5]
        for j in range(elit.shape[0]):
            s = elit[j]-rws
            e = elit[j]+rws
            
            if s >= 0 and e <= im.shape[1]:
                x_data = np.array(im[s:e,row-ws:row+ws,col-ws:col+ws])
                X.append(x_data)
                Y.append(label[i])
                Z.append([SE,row,col,j]) # [쥐번호, session번호, 세로, 가로, 프레임 위치]
                
    print(SE, 'x shape', X[0].shape, 'label', Y[0])
                
seed = 0      
random.seed(seed)
rix = random.sample(range(len(X)), len(X))
X = np.array(X)[rix]
Y = np.array(Y)[rix]
Z = np.array(Z)[rix]

print('samples distribution...', np.mean(Y, axis=0))

#roi_t4=[]; nonroi_t4=[]
#for roi in [np.where(Y==1)]:
#    roi_t4.append(np.mean(X[roi]))
#
#for roi in [np.where(Y==0)]:
#    nonroi_t4.append(np.mean(X[roi]))
#print('snr', np.mean(roi_t4)/np.mean(nonroi_t4))

msdata = {'X' : X, 'Y' : Y, 'Z' : Z, 'pathsave' : pathsave}
picklesavename = 'D:\\autoROI_CNN_result\\' + 'autoROI_rawdata.pickle'   
with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(msdata , f, pickle.HIGHEST_PROTOCOL)
    print(picklesavename, '저장되었습니다.')


import sys
sys.exit()
                
# In[] part2 - analysis
    
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import pandas as pd
import random
import pickle 
    
from datetime import datetime
from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
    
# 학습
# label 나누어서 time에 따른 시계열 뽑음
# 시계열 데이터 균일화
# 1차 모델 RNN 학습
# RNN으로 시계열을 없애고 동영상 -> 이미지로 변환
# score 매겨보고..

# CNN, 이미지 to 이미지
picklesavename = 'D:\\autoROI_CNN_result\\' + 'autoROI_rawdata.pickle'           
with open(picklesavename, 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)            

X = msdata_load['X']
Y = msdata_load['Y']
Z = msdata_load['Z']       
pathsave = msdata_load['pathsave'] 
## X 길이 통일, 전처리
#lensave = []
#for u in range(X.shape[0]):
#    lensave.append(X[u].shape[0])
#length = np.min(lensave)
#print('minimum frame length is...', np.min(lensave))
#X2=[]
#for u in range(X.shape[0]):
#    X2.append(X[u][:length])
#X = np.reshape(X2, (1, np.array(X2).shape[0], np.array(X2).shape[1])); del X2
# X[sequence segment, datanum, sequence legnth]

# training set list 뽑기
# [SE,se,row,col]
mouselist = list(set(Z[:,0]))

# RNN model 정의
msunit = 1 # 시계열 분할 정도
fn = 1 # 시계열 분할 확장 (나중에 n_in으로 통합되야.. 현재는 전혀 안쓰므로 남겨만 놓겠음)
inputsize = np.zeros(msunit *fn, dtype=int) 
for unit in range(msunit *fn):
    inputsize[unit] = X[unit].shape[1]
n_in =  1 # number of features
n_out = 2 # number of class
n_hidden = int(8 * 3) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 3) # fully conneted laye node 갯수 # 8
l2_rate = 0.25 # regularization 상수
dropout_rate1 = 0.20 # dropout late1
dropout_rate2 = 0.10 # dropout late2
lr = 1e-3 # learning rate
batch_size = 100

sequencesize = np.array(X).shape[1]
rowsize = np.array(X).shape[2]
colsize = np.array(X).shape[3]

seed = 1
def keras_setup():
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    
    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    input1 = []; [input1.append([]) for u in range(rowsize*colsize)]
    input2 = []; [input2.append([]) for u in range(rowsize*colsize)]
    
    for u in range(rowsize*colsize):
        input1[u] = keras.layers.Input(shape=(sequencesize, fn)) # 각 병렬 layer shape에 따라 input 받음
        input2[u] = Bidirectional(LSTM(n_hidden))(input1[u]) # biRNN -> 시계열에서 단일 value로 나감
        input2[u] = Dense(layer_1, kernel_initializer = init, activation='relu')(input2[u]) # fully conneted layers, relu
        input2[u] = Dropout(dropout_rate1)(input2[u]) # dropout
        input2[u] = Dense(layer_1, kernel_initializer = init, activation='relu')(input2[u]) # fully conneted layers, relu
        input2[u] = Dropout(dropout_rate1)(input2[u]) # dropout
    
    added = keras.layers.Add()(input2) # 병렬구조를 여기서 모두 합침
    merge1 = Dense(layer_1, kernel_initializer = init, activation='relu', \
                   kernel_regularizer=regularizers.l2(l2_rate))(added) # fully conneted layers, relu
    merge1 = Dropout(dropout_rate2)(merge1) # dropout
    merge1 = Dense(layer_1, kernel_initializer = init, activation='relu', \
                   kernel_regularizer=regularizers.l2(l2_rate))(merge1) # fully conneted layers, sigmoid
    merge2 = Dense(n_out, kernel_initializer = init, activation='sigmoid')(merge1) # fully conneted layers, relu
    merge2 = Activation('softmax')(merge2) # activation as softmax function
    
    model = keras.models.Model(inputs=input1, outputs=merge2) # input output 선언
    model.compile(loss='categorical_crossentropy', \
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), \
                  metrics=['accuracy']) # optimizer
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup()    
RESULT_SAVE_PATH = 'D:\\autoROI_CNN_result\\'    
initial_weightsave = RESULT_SAVE_PATH + 'initial_weight.h5'
model.save_weights(initial_weightsave)


# 학습시작
# Cross validation 구현, training
# In[]
i=0
for i in range(len(mouselist)):
    mouseNum = mouselist[i]
    
    dellist = np.where(Z[:,0]==mouseNum)[0] # [SE,row,col]

    tr_x = np.delete(X, dellist, 0)
    tr_y = np.delete(Y, dellist, 0)
    
    rix = random.sample(list(range(tr_x.shape[0])), int(tr_x.shape[0]/1000))
    tr_x = tr_x[rix]
    tr_y = tr_y[rix]
    
    tr_x2 = []
    for r in range(rowsize):
        for c in range(colsize):
            tmp = np.array(tr_x[:,:,r,c])
            tr_x2.append(np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1)))

    
    tr_y2=[]
    for sample in range(tr_y.shape[0]):
        if tr_y[sample] == 0:
            label = [1,0]
        elif tr_y[sample] == 1:
            label = [0,1]
            
        tr_y2.append(np.array(label))
    tr_y2 = np.array(tr_y2)

    # validation
#    valid_x = X[dellist]
#    valid_y = Y[dellist]
    
#    valid_x2=[]
#    for r in range(rowsize):
#        for c in range(colsize):
#            tmp = np.array(valid_x[:,:,r,c])
#            valid_x2.append(np.reshape(np.array(label), ()
    
#    valid_y2=[]
#    for sample in range(valid_y.shape[0]):
#        if valid_y[sample] == 0:
#            label = [1,0]
#        elif valid_y[sample] == 1:
#            label = [0,1]
#            
#        valid_y2.append(label)
#    valid_y2 = np.array(valid_y2)
#    
#    valid = (valid_x2, valid_y2)
#    
    # training
    model.fit(tr_x2, tr_y2, batch_size = batch_size, epochs = 100)





























































