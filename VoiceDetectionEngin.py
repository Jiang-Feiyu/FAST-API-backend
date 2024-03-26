
"""
The major use of VoiceDetectionEngine is to handle logics that process in the API.  

For example, get all audio file names, save audio are not related to the networking issue, 
so we can separate them.   

In case the function changes,  we need not to change the API - the API is the same, what changes is
just the process inside each function.  

In future, if we need to rewrite functions like save_audio, we only changes this file, instead of changing
in the API.
"""

def import_packages():
    # 标准库
    import sys
    import os
    import pathlib
    import pickle
    from datetime import datetime
    import wave
    from os import listdir
    from os.path import isfile, join
    from random import randint
    import re
    from random import randint

    # 第三方库
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.image as imgshow
    import numpy as np
    from scipy.io import wavfile
    from scipy.io.wavfile import write
    from IPython import display
    from colorama import Fore
    import numpy as np
    import concurrent.futures
    import soundfile as sf
    from scipy.signal import resample

    # TensorFlow 和 Keras
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # HDF5
    import h5py

    # 返回需要的模块
    return sys, os, pathlib, pickle, datetime, wave, listdir, isfile, join, randint, re, librosa, plt, imgshow, np, wavfile, write, display, Fore, tf, load_model, h5py, concurrent, sf, resample

# 调用函数来导入所有包
sys, os, pathlib, pickle, datetime, wave, listdir, isfile, join, randint, re, librosa, plt, imgshow, np, wavfile, write, display, Fore, tf, load_model, h5py, concurrent, sf, resample = import_packages()

dbtrain_basic=['000','111','222','333','444','555','666','777','888','999','AAA','BBB','CCC','DDD','EEE','FFF','GGG','HHH','III','JJJ','KKK','LLL','MMM','NNN','OOO','PPP','QQQ','RRR','SSS','TTT','UUU','VVV','WWW','XXX','YYY','ZZZ','!!!',',,,','+++','(((',')))','$$$','%%%','###','@@@','&&&']
print('Number of dbtrain_basic 3-word sentences: ', len(dbtrain_basic))

dbtrain=['1WX', '!RC', '2ST', '!(2', '2UT', 'VJ2', '01R', '($#', '$@#', '023', '234', 'OPQ', 'R23', '!$%', '($%', 'J(J', 'HWX', 'CD)', '!2,', 'O,+', 'OPM', 'VFI', 'RCD', 'YBA', '0(2', 'YEI', 'REI', '0EI', 'R$%', 'EFG', 'YJ2', '!,)', '!5L', '0O!', 'SKL', '!J2', 'JKL', '(2S', '02M', 'RAB', '2Y0', 'RHI', 'E(E', '!JL', 'M$%', '287', '(16', '0U2', '!RV', 'R,)', ',+)', 'R27', '!$#', '(2H', '0JW', '!CD', 'C97', '!O!', '1J2', '0JL', '!R%', '34)', 'OST', '!JW', '0RM', 'O$%', '012', '5ST', '@&+', '0(M', '2(2', '(12', '(27', '!EX', 'HFI', '(2T', '1AB', 'RET', '(26', '2R0', '1OP', '2R!', '!1R', '!12', '!UT', 'YPZ', 'YCD', 'O,)', 'VWX', '(1)', '2$#', '1ZP', '057', '!26', '(AB', '$&%', 'O$#', '97Q', '(,)', '(57', '0,+', '!27', '!R6', '1CW', 'S34', '1@&', 'V,+', 'OUT', '2AB', '(OP', ',&+', 'J97', '(1R', '(R6', 'VBA', 'RUW', '!(R', 'R,+', '0HI', '123', '(1H', '0OP', '97)', '2VX', '0R!', 'J67', '0R$', '027', '0RC', '!RZ', '!23', '597', '(UT', 'HFG', 'LEI', 'EWX', 'M$#', 'VJ2', 'K7Q', '1CA', '!R$', '1UT', '(2$', '1$#', '(1V', 'RMP', 'R5L', '(2L', '(1Y', 'KLQ', '(R,', 'O!2', '567', 'C34', '1OP', 'Y02', '!2R', 'MWX', '2OP', '0EX', 'RWX', '(ST', '2CD', 'R2L', '!U2', '97I', 'MFG', '(CD', 'S87', '!2$', '!ST', '2,+', '1MX', '0R2', '!HI', '(HI', 'RST', '0($', 'REX', 'RVX', '1,)', '!(M', 'R57', '0$#', 'V@&', '!EI', 'YFZ', '0UT', '!R2', '0J2', 'MFZ', 'OP2', '0RZ', '(2E', '1FZ', '0(R', '02$', 'OCD', '1ST', '2,)', '333', '444', '888', '999', 'GGG', 'KKK', 'NNN', 'QQQ', '@@@', '&&&']
dbtrain = dbtrain_basic + dbtrain

dbtest=['1EI', '0ST', '587', '(1S', 'R$#', '(2R', 'VFQ', '0AB', 'YHI', '(R0', '0JW', '87Q', 'Z($', 'M,)', '!1S', '!OP', 'ROP', '(R!', '(1E', 'LHI', 'OPV', '!JW', 'VY0', '0R,', 'MTQ', 'ZY0', 'RZP', 'VFZ', '0$%', 'RJ2', '$@&', '1CD', 'J87', '!2V', '2Y!', '026', ',@+', 'V$%', '(13', 'YWX', '(2M', '!R,', 'EFI', '$#%', 'J34', '02L', '87I', '0CD', 'O@&', '297']
commands= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','!',',','+','(',')','$','%','#','@','&']

#322 items; Sept 18, 2022
valid=[]
valid.extend(['012', '01S', '01R', '057', '05L', '0JL', '0J2', '0JW', '0EI', '0HI', '0CD', '02T', '02$', '02,', '02V', '027', '023', '026', '02L', '02M', '02R', '02E', '0RZ', '0RM', '0RV', '0R6', '0R!', '0R$', '0R#', '0R%', '0R2', '0RC', '0R,', '0EX', '0OP', '0O!', '0ST', '0UT', '0U2', '0AB', '0,+', '0,)', '0(R', '0(2', '0($', '0(M', '0$%', '0$#', '0JW', '1CD', '1FZ', '1CW', '1FG', '1EI', '1WX', '1UT', '1ST', '123', '1AB', '1OP', '1HI', '1ZP', '1OP', '1J2', '1@&', '1$#', '1$%', '1,)', '1,+', '1MX', '1CA', '234', '2CD', '2AB', '2KL', '2ST', '2UT', '2OP', '297', '287', '2VX', '2Y0', '2Y!', '2(2', '2,+', '2,)', '2$%', '2$#', '2R0', '2R!', '34)', '567', '587', '597', '5KL', '5ST', '5J2', '5ST', '5UT', '87Q', '87I', '97Q', '97I', '97)', 'C34', 'C97', 'CD)', 'EFG', 'EFI', 'EWX', 'EFN', 'E(E', 'HFG', 'HFI', 'HWX', 'HFZ', 'J34', 'J67', 'JKL', 'J87', 'J97', 'J(J', 'KLQ', 'K7Q', 'LEI', 'LHI', 'MFG', 'MFN', 'MWX', 'MTQ', 'MFZ', 'M$#', 'M$%', 'M,)', 'M,+', 'OPQ', 'OP2', 'OST', 'OUT', 'OCD', 'OAB', 'OPV', 'OPM', 'O02', 'O!2', 'O,+', 'O,)', 'O$#', 'O$%', 'O@&', 'R02', 'RAB', 'RCD', 'ROP', 'RST', 'RUT', 'RWX', 'RZP', 'RMQ', 'RVI', 'RVX', 'R23', 'RJ2', 'R57', 'R5L', 'RUW', 'REI', 'RHI', 'RFG', 'RMP', 'R27', 'R2L', 'RET', 'REX', 'R!2', 'R,)', 'R,+', 'R$%', 'R$#', 'R(R', 'R(,', 'R($', 'ST6', 'SKL', 'S34', 'S67', 'S87', 'S97', 'SUT', 'VWX', 'VJ2', 'VFZ', 'VY0', 'VBA', 'VFI', 'VFQ', 'VJ2', 'V@&', 'V$%', 'V$#', 'V,)', 'V,+', 'YWX', 'YFZ', 'YBA', 'YJ2', 'YPZ', 'Y02', 'YAB', 'YCD', 'YEI', 'YHI', 'ZYP', 'ZY0', 'ZY!', 'Z(Y', 'Z($', 'Z(,', '!12', '!1S', '!1R', '!57', '!5L', '!JL', '!J2', '!JW', '!EI', '!HI', '!CD', '!2T', '!2$', '!2,', '!2V', '!27', '!23', '!26', '!2L', '!2M', '!2R', '!2E', '!RZ', '!RM', '!RV', '!R6', '!R0', '!R$', '!R#', '!R%', '!R2', '!RC', '!R,', '!EX', '!OP', '!O!', '!ST', '!UT', '!U2', '!AB', '!,+', '!,)', '!(R', '!(2', '!($', '!(M', '!$%', '!$#', '!JW', '(,+', '(,)', '($%', '($#', '(R,', '(R$', '(R!', '(R0', '(57', '(16', '(12', '(1Y', '(1,', '(1O', '(1V', '(1$', '(1)', '(13', '(1M', '(1H', '(1R', '(1S', '(1E', '(OP', '(R2', '(AB', '(CD', '(EI', '(HI', '(R6', '(ST', '(UT', '(26', '(2T', '(2Y', '(2$', '(2,', '(2O', '(2V', '(23', '(27', '(2L', '(2M', '(2H', '(2R', '(2S', '(2E', '$@#', '$&%', '$#%', '$@&', ',+)', ',@+', ',&+', '@&+'])

def DisplayResFull(full_list):
  print("Speech_Recognition.py: DisplayResFull(): NOT USED")

  DISPLAYFigure = 0   # Cancel the display of detection results
  if (DISPLAYFigure == 1):
    if (len(full_list) == 1):
      fig = plt.figure(figsize=(4, 3), dpi=80)
      showresID_tmp = full_list[0]
      for i in range(0,3):
        k = showresID_tmp[i]
        plt.subplot(1,3,i+1)  #int(ID_length)
        img =imgshow.imread('./public/image/JPG/'+str(k)+'.JPG')
        plt.imshow(img)
        plt.title('ID_'+str(showresID_tmp))
        plt.axis('off')
      plt.show()
	    
    if (len(full_list) > 1 and len(full_list) < 7):
      fig = plt.figure(figsize=(12, 4), dpi=80)
      for j in range(0,len(full_list)):
        showresID_tmp = full_list[j]
        empty_space=j
        for i in range(0,3):
          k = showresID_tmp[i]
          plt.subplot(3,12,3*j+i+1+empty_space)  #int(ID_length)
          img =imgshow.imread('./public/image/JPG/'+str(k)+'.JPG')
          plt.imshow(img)
          plt.title('ID_'+str(showresID_tmp))
          plt.axis('off')
      plt.show()
    if (len(full_list) > 6):
	    print('Error: DisplayResFull has more than 6 items.')
     
# 检验h5文件是否损坏
def check_hdf5_file(file_path):
    try:
        # 打开 HDF5 文件
        with h5py.File(file_path, 'r') as f:
            # 检查文件中是否包含模型的关键组或数据集
            # 你可以根据实际情况修改下面的检查条件
            if 'model_weights' in f.keys() or 'model_config' in f.keys():
                print("HDF5 文件有效，模型文件没有损坏。")
            else:
                print("HDF5 文件不包含模型数据，可能已损坏。")
    except Exception as e:
        print("加载 HDF5 文件时出现错误：", e)

# 检验pickle文件是否损坏    
def check_pickle_file(file_path):
    with open(file_path, 'rb') as fp:
        try:
            data = pickle.load(fp)
            print("Loaded pickle file successfully:", data)
            return data
        except Exception as e:
            print("Error loading pickle file:", e)
            return None

# 并发加载模型
def load_model_task(model_path):
    print("model loading from path:", model_path)
    return load_model(model_path)

# 并发加载数据
def load_pickle_file(pickle_path):
    with open(pickle_path, 'rb') as fp:
        return pickle.load(fp)

# 加载模型和数据
def load_models_and_pickles(model_paths, pickle_paths):
    # Load models
    with concurrent.futures.ThreadPoolExecutor() as executor:
        models = list(executor.map(load_model_task, model_paths))
    
    # Load pickles
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pickle_data = list(executor.map(load_pickle_file, pickle_paths))
    
    return models, pickle_data

# A function to read an audio .wav file and shows the waveform
def showAudio(file_name):
    waveform1 = []  # 初始化波形数组
    print("Speech_Recognition.py: showAudio() version Jan 1,2024 ")  # 打印版本信息
    audio_binary = tf.io.read_file(file_name)  # 读取音频文件的二进制数据
    audio, sampleR = tf.audio.decode_wav(audio_binary)  # 解码音频文件，并获取音频数据和采样率
    print("next&&&&&&&&&&")
    
    if (sampleR != 48000):  # 如果采样率不是48000（标准采样率）
        print("showAudio(): Conversion is needed. To start now.")  # 打印提示信息，需要进行转换
        yy, sr = sf.read(file_name)
        yy_resampled = resample(yy, int(len(yy) * (48000 / sr))) # 对音频数据进行采样率转换为48000Hz
        sf.write(file_name, yy_resampled, 48000)
        audio_binary = tf.io.read_file(file_name)  # 重新读取转换后的音频文件
        audio, sampleR = tf.audio.decode_wav(audio_binary)  # 解码转换后的音频文件
        print("showAudio(): Conversion is completed.")  # 打印转换完成的提示信息

    waveform2 = audio.numpy()  # 将音频数据转换为numpy数组
    if (waveform2.shape[1] == 1):  # 如果音频为单声道
        waveform1 = waveform2  # 直接将音频数据赋值给波形数组
    else:  # 如果音频为立体声
        waveform1 = waveform2[:, 1]  # 取出立体声音频中的右声道，转换为单声道
        
    return waveform1  # 返回处理后的波形数据数组

def resolve(waveform1, xcoords):
    """
    函数resolve用于处理音频波形的坐标,并根据条件对其进行调整。

    参数：
        waveform1 (array): 音频波形数据。
        xcoords (list): 包含音频节拍坐标的列表，格式为 [start1, end1, start2, end2, ...]。

    返回：
        list: 调整后的音频节拍坐标列表。

    - 如果节拍数量为3,则直接使用原始坐标
    - 如果节拍数量为2,则根据长度比较来调整坐标
    - 如果节拍数量为1,则在节拍区间内寻找最小值，并将其添加到调整后的坐标列表中
    - 如果节拍数量为4、5或6,则选择其中最大的3个节拍坐标,并将其添加到调整后的坐标列表中
    - 如果节拍数量不符合以上条件，则打印错误信息; 如果传入的坐标列表为空，则打印错误信息
    """
    print("Speech_Recognition.py: resolve()")
    xcoordsW = []

    if len(xcoords) != 0:
        num_sections = len(xcoords) // 2  # 计算节拍数量

        if num_sections in [4, 5, 6]:
            # 处理节拍数量为4、5或6的情况
            print(f'number of AUDIO SECTION is {num_sections}. ')
            gaps = [xcoords[i + 1] - xcoords[i] for i in range(0, len(xcoords), 2)]
            index_tmp = np.argsort(gaps)
            selected_index = index_tmp[:3] if num_sections != 6 else index_tmp[3:]
                
            # 根据选定的索引构建新的坐标列表
            for section_index in selected_index:
                xcoordsW += [xcoords[section_index * 2], xcoords[section_index * 2 + 1]]

            print('xcoordsW', xcoordsW)
            
        elif num_sections in [1, 2, 3]:
            if num_sections == 2:
                print('number of AUDIO SECTION is TWO. ')
                aa, bb = xcoords[0], xcoords[-1]
                length_1st = xcoords[1] - xcoords[0]
                length_2nd = xcoords[3] - xcoords[2]

                if length_1st > length_2nd:
                    middle_1st_index = xcoords[0] + int(0.5 * length_1st)
                    tmpVal1 = findmin(waveform1, int(middle_1st_index - length_1st * 0.25),
                                      int(middle_1st_index + length_1st * 0.25))
                    xcoordsW = [aa, tmpVal1, tmpVal1, xcoords[1], xcoords[2], bb]
                    print('xcoordsW', xcoordsW)
                else:
                    middle_2nd_index = xcoords[2] + int(0.5 * length_2nd)
                    tmpVal2 = findmin(waveform1, int(middle_2nd_index - length_2nd * 0.25),
                                      int(middle_2nd_index + length_2nd * 0.25))
                    xcoordsW = [aa, xcoords[1], xcoords[2], tmpVal2, tmpVal2, bb]
                    print('xcoordsW', xcoordsW)

            elif num_sections == 1:
                print('number of AUDIO SECTION is ONE. ')
                aa, bb = xcoords[0], xcoords[1]
                dd = int(bb - aa)
                tmpVal1 = findmin(waveform1, int(aa + dd / 3 - dd / 9), int(aa + dd / 3 + dd / 9))
                tmpVal2 = findmin(waveform1, int(aa + dd * 2 / 3 - dd / 9), int(aa + dd * 2 / 3 + dd / 9))
                xcoordsW = [aa, tmpVal1, tmpVal1, tmpVal2, tmpVal2, bb]
                print('xcoordsW', xcoordsW)

            elif num_sections == 3:
                xcoordsW = xcoords

        else:
            print('ERROR: Unexpected number of audio sections')

    else:
        print('ERROR: Length of xcoords is zero in resolve()')

    return xcoordsW

def append_new_line(file_name, text_to_append):
    # Append given text as a new line at the end of file
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
    file_object.close()
    
def convertIndextoID(dbindex):
    tmpresult = []
    for i in range(len(dbindex)):
        tmpresult.append(commands[int(dbindex[i])])        
    return tmpresult

# To get back the integer index [0,45] value from character, and output the index
def findIndex(tmpresID):
    k1=tmpresID[0]
    k2=tmpresID[1]
    k3=tmpresID[2]
    for i in range(46):
        if (commands[i] == k1):
            out1 = i
        if (commands[i] == k2):
            out2 = i   
        if (commands[i] == k3):
            out3 = i
    resultout = [out1 , out2, out3]
    return resultout

# Return only Valid Items based on the valid list
def checkValid(cOutword,cOutprob):
    Outword = [ ]
    Outprob = [ ]
    for i in range(len(cOutprob)):
        wordItem = cOutword[i]
        probItem = cOutprob[i]
        if wordItem in valid:
            Outword.append(wordItem)
            Outprob.append(probItem)
    return Outword, Outprob

def findmin(waveform,start_index,end_index):
  print("Speech_Recognition.py: findmin()")
  waveformN=1.5*np.zeros(end_index - start_index,dtype=float)
  # To get the largest within the next 1000 data
  for i in range(0,len(waveformN)):
    waveformN[i] = np.max(np.abs(waveform[i+start_index:i+start_index+1000]))
    
  save_x = -1
  tmp = waveformN[0]
  # Next, we want to find the lowest within the section
  for i in range(0, len(waveformN)):
    if (waveformN[i] < tmp):
      tmp = waveformN[i]
      save_x = i 
  answer_x = save_x + start_index
  return answer_x

def newevaluateResult ( resIDmodel1 , resIDmodel2, DBwordID, DBprob, DBwordID1, DBprob1 ):   
    resIndex1 = findIndex(resIDmodel1)
    resIndex2 = findIndex(resIDmodel2)
    print("resIndex1", resIndex1, "resIndex2", resIndex2)
    # Get all the valid possible combinations (via the three FOR loops) and associated probabilities
    vOutword1, vOutprob1 = getValid(resIndex1,DBwordID1,DBprob1,0) #model1:cnn_yang1
    vOutword, vOutprob = getValid(resIndex2,DBwordID,DBprob,0) #model2: model6n
    # Combine the two valid lists together
    tOutword = vOutword+vOutword1
    tOutprob = vOutprob+vOutprob1
    print("tOutword", tOutword, "tOutprob", tOutprob)
    # Combine with the two identified words
    if resIDmodel1 in valid:
        tOutword.append(resIDmodel1)
        tOutprob.append(1)
    if resIDmodel2 in valid:
        tOutword.append(resIDmodel2)
        tOutprob.append(1)
    sortOutword, sortOutprob = getSort(tOutword,tOutprob,0)
    resultword, resultprob = removeDuplicate(sortOutword, sortOutprob,0)
    resultwordtop3, resultprobtop3 = getTop3(resultword, resultprob) # only top two is needed

    return resultwordtop3, resultprobtop3

def getTop3(resultword, resultprob):
    if (len(resultword) <= 3 ):
        resultwordtop3 = resultword
        resultprobtop3 = resultprob
    else:
        resultwordtop3 = [ resultword[0], resultword[1], resultword[3] ]
        resultprobtop3 = [ resultprob[0], resultprob[1], resultprob[3] ]
    return resultwordtop3, resultprobtop3

def removeDuplicate(inword, inprob, printindex):
    # remove duplicated from list 
    resultword = [] 
    resultprob = []
    for i in range(len(inword)):
        kk = inword[i]
        if kk not in resultword: 
            resultword.append(kk)
            resultprob.append(inprob[i]) 
    if (printindex == 1):
        print(resultword)
        print(resultprob)
    return resultword, resultprob

# Convert the waveform to have 72000 (1.5 seconds) data points
def covert_72000(waveform):  # Standardize to length of 72000; 1.5 SECONDS
  # if more than 72000, trim to 72000
  if (len(waveform) >= 72000):
    waveform = waveform[0:72000]
  # Padding for files with less than 72000 samples
  zero_padding = tf.zeros([72000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the same length
  waveform = tf.cast(waveform, tf.float32)  # NO NEED TO DOUBLE CAST
  equal_length = tf.concat([waveform, zero_padding], 0)
  return equal_length

def get_new_spectrogram(filename,samplerate,SHOW):
    print("Speech_Recognition.py: get_new_spectrogram()")
    #signal, sr = librosa.load(wavfileinput, sr=samplerate)
    signal, sr = librosa.load(filename, sr=samplerate)

    signal = covert_72000(signal)    
    signal = signal.numpy()  # need this after conversion
    #print('len(signal) after cast = ',len(signal),type(signal))
    
    # Step to Convert audio waveform to spectrogram
    # Spectrogram only make use of 256 rows, for frequencies to around 10K Hz.
    X = abs(librosa.stft(signal,n_fft=1024))
    # print('X.shape ',X.shape)  # gives (513,282) after the wav is 1.5 seconds
    X = X[0:256,:]  # gives (256,282)

    if (SHOW ==1):
      # Figure 1 : waveform
      plt.figure(figsize=(12, 4))
      librosa.display.waveshow(signal, sr=sr)
      plt.title('waveform : '+filename)
      plt.show()
      # Figure 2 : X
      plt.figure(figsize=(12, 4))
      librosa.display.specshow(X, sr=sr, y_coords=None, x_axis='time', y_axis=None)  #y_axis='hz'
      y_ticks=[1,15,50,100,155,255]
      y_labels=['43Hz','645Hz','2153Hz','4306Hz','6675Hz','10982Hz']
      plt.yticks(ticks=y_ticks, labels=y_labels)
      plt.colorbar()
      plt.title('X : '+filename)
      plt.show()
      # Figure 3 : Xdb
      Xdb = librosa.amplitude_to_db(X)
      plt.figure(figsize=(12, 4))
      librosa.display.specshow(Xdb, sr=sr, y_coords=None, x_axis='time', y_axis=None)  #y_axis='hz'
      y_ticks=[1,15,50,100,155,255]
      y_labels=['43Hz','645Hz','2153Hz','4306Hz','6675Hz','10982Hz']
      plt.yticks(ticks=y_ticks, labels=y_labels)
      plt.colorbar()
      plt.title('Xdb : '+filename)
     
      print('max/min(X): ',np.max(X), np.min(X))
      print('max/min(Xdb): ', np.max(Xdb),np.min(Xdb))
      plt.show()
        
    return X

# For each detected word (in resIndex), get the potential words from DBwordID with DBprob
# get all possible combinations (via the three FOR loops) and associated probabilities
# keep only the valid items and output
def getValid(resIndex,DBwordID,DBprob,printindex):
    word1 = resIndex[0]
    word2 = resIndex[1]
    word3 = resIndex[2]
    
    DBword1ID = DBwordID[word1]
    DBword2ID = DBwordID[word2]
    DBword3ID = DBwordID[word3]
    DBword1prob = DBprob[word1]
    DBword2prob = DBprob[word2]
    DBword3prob = DBprob[word3]
    
    Outword = [ ]
    Outprob = [ ]
    for i in range(len(DBword1ID)):
        for j in range(len(DBword2ID)):
            for k in range(len(DBword3ID)):
                tmpword = DBword1ID[i]+DBword2ID[j]+DBword3ID[k]
                tmpprob = (DBword1prob[i]+DBword2prob[j]+DBword3prob[k])/3
                Outword.append(tmpword)
                Outprob.append(tmpprob)

    vOutword,vOutprob = checkValid(Outword,Outprob)
    if (printindex == 1):
        print(Outword)
        print(Outprob)
        print(vOutword)
        print(vOutprob)
    
    return vOutword, vOutprob

# Output is from the largest to the smallest
def getSort(tOutword,tOutprob,printindex):
    sortAnswer = np.argsort(tOutprob)
    sortOutword = [ ]
    sortOutprob = [ ]
    for i in range(len(tOutprob)):
        sortOutword.append(tOutword[int(sortAnswer[len(tOutprob)-i-1])])
        sortOutprob.append(tOutprob[int(sortAnswer[len(tOutprob)-i-1])])
    if (printindex == 1):
        print(sortOutword)
        print(sortOutprob)
    return sortOutword, sortOutprob

def get_waveform_section_v2(file_name, Z, SHOW):
  print("Speech_Recognition.py: get_waveform_section_v2()")
  audio_binary = tf.io.read_file(file_name)
  audio , sampleR = tf.audio.decode_wav(audio_binary)
  waveform2=audio.numpy()  # may be mono or stereo
  if (waveform2.shape[1] == 1):
    waveform1=waveform2[:,0]
  else:
    waveform1=waveform2[:,1] #stereo changed to mono

  waveformN= np.zeros((len(waveform1)),dtype=float)
  for i in range(0,(len(waveform1)-Z)):
    tmpVal = np.max(np.abs(waveform1[i:i+Z]))
    # threshold to 0.035 from 0.05; helps when sound is low; on May 24, 2022
    if (tmpVal > 0.03): # threshold to 0.03 ; helps when sound is low; on Dec 28, 2022
      waveformN[i]=0.8
    else:
      waveformN[i]=0

  for i in range(len(waveform1)-Z,len(waveform1)):
    waveformN[i]=0
  
  # Check for suitable audio sections
  audioLength=[]
  audioLength_section=[]
  xcoords = []
  colors = []
  Astart=-1
  Astop=-1
  for i in range(len(waveform1)-10):
    if (waveformN[i] == 0 and waveformN[i+1] == 0.8):
      Astart=i
    if (waveformN[i] == 0.8 and waveformN[i+1] == 0):
      Astop=i
      if (Astart != -1):
        Aduration = Astop - Astart
        # print('Aduration (expected to longer than 0.2sec or 9000) = ',Aduration)  
        # changed to 11500 on Sept 29, 2022
        # Dec 28, 2022 : Astop + 5000 (to get better ending of a .wav)
        if (Aduration > 11500):
          audioLength = audioLength + ([Aduration])
          xcoords = xcoords +([Astart , Astop+5000])
          Actual_duration = Astop + 5000 - Astart
          if (Actual_duration > 72000):
            print('*********************** warning LONG audio section ******* = ', Actual_duration, ' fileanme ',file_name)
          audioLength_section = audioLength_section + ([Actual_duration])
          colors = colors + (['r','g'])
          Astart = -1
          Astop = -1
  print('length of audioLength ',audioLength_section)

  return xcoords

def run_interpret_audio(user_id: int):
    print("VoiceDetectionEngine.py: run_interpret_audio()")

    AIanswer = execute_interpret(user_id)
    return AIanswer

# 主函数 - 文字化输入内容
def execute_interpret(file_index: int):

    cnn_yang1=load_model('./models/Models-LHo-20230111/cnn_yang1.h5')
    model6n=load_model('./models/Models-LHo-20230111/model6n.h5')
  
    model_paths = [
            './models/Models-LHo-20230111/cnn_yang1.h5',
            './models/Models-LHo-20230111/model6n.h5'
        ]
    pickle_paths = [
            './models/Models-LHo-20230111/saveDBindex',
            './models/Models-LHo-20230111/saveDBindex1',
            './models/Models-LHo-20230111/saveDBprob',
            './models/Models-LHo-20230111/saveDBprob1',
            './models/Models-LHo-20230111/saveDBwordID',
            './models/Models-LHo-20230111/saveDBwordID1'
    ]
    
    with open('./models/Models-LHo-20230111/saveDBindex','rb') as fp:
        DBindex = pickle.load(fp)

    with open('./models/Models-LHo-20230111/saveDBprob','rb') as fp:
        DBprob = pickle.load(fp)

    with open('./models/Models-LHo-20230111/saveDBwordID','rb') as fp:
        DBwordID = pickle.load(fp)

    with open('./models/Models-LHo-20230111/saveDBindex1','rb') as fp:
        DBindex1 = pickle.load(fp)

    with open('./models/Models-LHo-20230111/saveDBprob1','rb') as fp:
        DBprob1 = pickle.load(fp)
		
    with open('./models/Models-LHo-20230111/saveDBwordID1','rb') as fp:
        DBwordID1 = pickle.load(fp)
    
    
    # Load models and pickles
    models, pickle_data = load_models_and_pickles(model_paths, pickle_paths)
    
    LoopContinue = 1
    index = 0
    ndex = 0
    tmp1 = 0
    file_chosen = ''
    xcoordsW = []
    
    # print("file_index", file_index)
    
    mypath = './Data/'+str(file_index)+'.wav'
    print("mypath is: ", mypath)
    if os.path.exists(mypath):
        waveform1 = showAudio(mypath)
        print(waveform1)
        xcoords = get_waveform_section_v2(mypath, 2500, 1)

        # print('xcoords = ', xcoords)

        if len(waveform1) != 0 and xcoords != []:
            xcoordsW = resolve(waveform1, xcoords)
            print('#音频片段(xcoordsW)= ', len(xcoordsW) / 2, '  索引', index)

            if xcoordsW == []:
                return "..."

            if int(len(xcoordsW) / 2) == 3:
                Model1_Res = ''
                Model2_Res = ''

                for ii in range(0, 3):
                    word = waveform1[xcoordsW[2 * ii]:xcoordsW[2 * ii + 1]]
                    samplerate = 48000
                    bb = ii + 1

                    filenameW = './RecogData/Iword_' + str(bb) + '_.wav'

                    # print('DBwordfile ', bb, ':', filenameW)
                    write(filenameW, samplerate, word)

                    word_spectrogram = get_new_spectrogram(filenameW, samplerate, 0)
                    predict_data_array = np.array(word_spectrogram, dtype=float).reshape(1, 256, 282, 1)
                    y_pred1 = np.argmax(cnn_yang1.predict(predict_data_array), axis=1)
                    y_pred2 = np.argmax(model6n.predict(predict_data_array), axis=1)
                    tmp_Model1 = commands[int(y_pred1)]
                    Model1_Res = Model1_Res + tmp_Model1[0]
                    tmp_Model2 = commands[int(y_pred2)]
                    Model2_Res = Model2_Res + tmp_Model2[0]

                print(' Models: ', Model1_Res, ' ; ', Model2_Res, ' ; ')

                FOUNDLIST, FOUNDPROB = newevaluateResult (Model1_Res, Model2_Res, DBwordID, DBprob, DBwordID1, DBprob1)

                print('FOUNDLIST:', FOUNDLIST)

                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
                append_new_line('VD8_FullRecord.txt', '-----------------------' + dt_string + '-----------------------')
                append_new_line('VD8_FullRecord.txt', mypath)
                abc = FOUNDLIST + [' ; '] + FOUNDPROB
                mystring = ' '.join(map(str, abc))
                append_new_line('VD8_FullRecord.txt', '  result= ' + mystring)

                print('--------------------------------------------------------------------------------------')

                if len(FOUNDLIST) == 0:
                    print('无法完全识别语音。可能是以下内容之一：')
                    # DisplayResFull([Model1_Res, Model2_Res])
                    COMBINELIST = "..." + Model1_Res + Model2_Res
                    return COMBINELIST
                else:
                    if len(FOUNDLIST) == 1:
                        print('最可能的音频：')
                        # DisplayResFull(FOUNDLIST)
                        return FOUNDLIST[0]
                    if len(FOUNDLIST) == 2:
                        print('两个可能的音频：')
                        # DisplayResFull(FOUNDLIST)
                        COMBINELIST = FOUNDLIST[0] + FOUNDLIST[1]
                        return COMBINELIST
                    if len(FOUNDLIST) == 3:
                        print('三个可能的音频：')
                        # DisplayResFull(FOUNDLIST)
                        COMBINELIST = FOUNDLIST[0] + FOUNDLIST[1] + FOUNDLIST[2]
                        return COMBINELIST
            else:
                print('音频片段的数量不是3。错误!')
                return "..."
        else:
            print('保护不良录音的情况。')
            return "..."
    else:
        print(f"file {file_index} doesn't exist")
        return None