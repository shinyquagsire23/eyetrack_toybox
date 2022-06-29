from os import walk
import random

import tensorflow as tf
import numpy as np
import time

import decord as de

from globals import *

fpath_bad = ["DikablisT_22_10.mp4", "DikablisT_2_2.mp4", 
"DikablisT_23_5.mp4",
"DikablisT_20_12.mp4",
"DikablisT_20_5.mp4",
"DikablisT_23_3.mp4",
"DikablisT_24_5.mp4",
"DikablisT_11_3.mp4",
"DikablisT_22_5.mp4",
"DikablisT_24_7.mp4",
"DikablisT_24_8.mp4",
"DikablisT_22_9.mp4",
"DikablisT_23_6.mp4",
"DikablisT_22_8.mp4",
"DikablisT_21_7.mp4",
"DikablisT_23_7.mp4",
"DikablisT_20_6.mp4",
"DikablisT_24_11.mp4",
"DikablisT_22_7.mp4",
"DikablisT_23_8.mp4",
"DikablisT_1_9.mp4",
"DikablisT_2_9.mp4",
"DikablisT_17_1.mp4",
"DikablisT_27_3.mp4",
"DikablisSA_31_1.mp4",
"DikablisT_8_12.mp4",
"DikablisT_19_12.mp4",
"DikablisT_20_13.mp4",


"DikablisT_14_5.mp4",
"DikablisT_6_3.mp4",
"DikablisT_18_7.mp4",
"DikablisT_19_8.mp4",
"DikablisT_13_5.mp4",
"DikablisT_21_6.mp4",
"DikablisT_2_5.mp4",
"DikablisT_20_9.mp4",
"DikablisT_7_12.mp4",
"DikablisT_12_7.mp4",
"DikablisT_9_4.mp4",
"DikablisT_1_7.mp4",
"DikablisT_19_5.mp4",
"DikablisT_6_10.mp4",
]

def data_serve_init():
    global fpath_list, fpath_list_orig

    fpath_list_pre = next(walk(Globals.DIKABLIS_VIDEOS_ROOT), (None, None, []))[2]  # [] if no file

    fpath_list = [x for x in fpath_list_pre if x not in fpath_bad]
    #fpath_list = [x for x in fpath_list_pre if x not in fpath_bad and "DikablisT" in x]

    fpath_list_orig = fpath_list.copy()

    random.shuffle(fpath_list)

def data_serve_list():
    global fpath_list

    return fpath_list

def read_landmarks(fpath, frame_w=1.0, frame_h=1.0):
    rows = []
    with open(fpath, newline='') as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break

            split = line.split(";")
            row = {}
            row["FRAME"] = int(split[0])
            row["AVG_INACCURACY"] = float(split[1])
            row["LM_X"] = []
            row["LM_Y"] = []
            for i in range(2, len(split)-1, 2):
                row["LM_X"] += [float(split[i]) / frame_w]
            for i in range(3, len(split)-1, 2):
                row["LM_Y"] += [float(split[i]) / frame_h]
            
            #row = dict([a.replace(' ', '_'), float(x) if a != "FRAME" else int(x)] for a, x in row.items())
            avg_x = 0.0
            for x in row["LM_X"]:
                avg_x += x
            avg_x /= float(len(row["LM_Y"]))

            avg_y = 0.0
            for y in row["LM_Y"]:
                avg_y += y
            avg_y /= float(len(row["LM_Y"]))

            for i in range(0, len(row["LM_X"])):
                row["LM_X"][i] = avg_x
                row["LM_Y"][i] = avg_y
            

            row["LM_X"] = row["LM_X"][:Globals.NETWORK_OUTPUT_SIZE//2]
            row["LM_Y"] = row["LM_Y"][:Globals.NETWORK_OUTPUT_SIZE//2]

            rows += [row]
    return rows

def read_iris_eli(fpath, frame_w=1.0, frame_h=1.0):
    rows = []
    with open(fpath, newline='') as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break

            split = line.split(";")
            row = {}
            row["FRAME"] = int(split[0])
            row["ANGLE"] = float(split[1])
            row["CENTER_X"] = float(split[2]) / float(frame_w)
            row["CENTER_Y"] = float(split[3]) / float(frame_h)
            row["WIDTH"] = float(split[4]) / float(frame_w)
            row["HEIGHT"] = float(split[5]) / float(frame_h)

            rows += [row]

        last_positive_y = 0.0
        last_positive_x = 0.0
        for j in range(0, len(rows)):
            #print (j)
            

            if rows[j]["CENTER_X"] < 0.0:
                rows[j]["CENTER_X"] = last_positive_x
            else:
                last_positive_x = rows[j]["CENTER_X"]

        
        
            if rows[j]["CENTER_Y"] < 0.0:
                rows[j]["CENTER_Y"] = last_positive_y
            else:
                last_positive_y = rows[j]["CENTER_Y"]
        
        last_x = rows[0]["CENTER_X"]
        last_y = rows[0]["CENTER_Y"]
        for j in range(0, len(rows)):
            rows[j]["CENTER_X_VEL"] = rows[j]["CENTER_X"] - last_x
            rows[j]["CENTER_Y_VEL"] = rows[j]["CENTER_Y"] - last_y

            last_x = rows[j]["CENTER_X"]
            last_y = rows[j]["CENTER_Y"]
    return rows

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 batch_size=Globals.TRAIN_BATCH_SIZE,
                 sel_idx_shift=0,
                 shuffle=True,
                 validation=False):
        
        self.batch_size = batch_size
        self.fpath_sel_idx = sel_idx_shift
        self.use_fpath_selector = False
        self.validation = validation

        self.frame_h = Globals.TRAIN_INPUT_SHAPE[1]
        self.frame_w = Globals.TRAIN_INPUT_SHAPE[2]
        self.frame_cnt = Globals.TRAIN_MAX_SEQS
        self.current_frame_y = 0
        #self.input_size = (self.frame_w, self.frame_h, 1)

        #if self.fpath is None:
        self.use_fpath_selector = True

        self.fpath_sel_idx = -1
        self.do_next_video()
        self.init_for_file()
    
    def do_next_video(self):

        self.fpath_sel_idx += 1
        if self.fpath_sel_idx >= len(fpath_list):
            self.fpath_sel_idx = 0

        
    def init_for_file(self):
        global fpath_list, fpath_list_orig
        self.video_fpaths = [""] * self.batch_size
        self.landmarks_fpaths = [""] * self.batch_size
        self.eli_fpaths = [""] * self.batch_size
        self.videos = [None] * self.batch_size
        self.eli_dicts = [None] * self.batch_size
        self.cur_frame = [0] * self.batch_size

        if len(fpath_list) < self.batch_size:
            fpath_list = fpath_list_orig.copy()
            random.shuffle(fpath_list)

        for i in range(0, self.batch_size):
            while True:
                self.video_fpaths[i] = '/Volumes/server_storage/training_data/Dikablis/VIDEOS/' + fpath_list.pop()
                self.landmarks_fpaths[i] = self.video_fpaths[i].replace("VIDEOS", "ANNOTATIONS") + "iris_lm_2D.txt"
                self.eli_fpaths[i] = self.video_fpaths[i].replace("VIDEOS", "ANNOTATIONS") + "iris_eli.txt"
        
                if Globals.TRAIN_ON_ANNOTATED:
                    self.video_fpaths[i] = self.video_fpaths[i].replace("VIDEOS", "ANNOTATIONS") + "iris_seg_2D.mp4"


                try:
                    #self.videos = None
                    cap_start = time.time()
                    #self.cap = [cv2.VideoCapture(self.fpath)]
                    shape = (Globals.TRAIN_MAX_SEQS, Globals.TRAIN_INPUT_SHAPE[1], Globals.TRAIN_INPUT_SHAPE[2], Globals.TRAIN_INPUT_SHAPE[3])
                    vid_list = [self.video_fpaths[i]]

                    # Play some samples at 2x speed
                    '''interval = 0
                    if random.random() < 10:
                        interval = 1'''
                    interval=0

                    vl = de.VideoLoader(vid_list, ctx=de.cpu(0), shape=shape, interval=interval, skip=0, shuffle=2)
                    
                    self.videos[i] = vl.next()
                    del vl

                    print (time.time() - cap_start, "to init", self.video_fpaths[i])

                except Exception as e:
                    print ("Failed to load", self.video_fpaths[i])
                    print (e)
                    #self.do_next_video()
                    continue
                
                try:
                    dict_start = time.time()
                    self.eli_dicts[i] = read_iris_eli(self.eli_fpaths[i], self.frame_w, self.frame_h)
                    print (time.time() - dict_start, "to read iris_eli")
                except Exception as e:
                    print ("Failed to load iris_eli", self.video_fpaths[i])
                    print (e)
                    #self.do_next_video()
                    continue

                #print (self.fpath_sel_idx)
                break
        

        self.current_frame_y = 0

        #self.landmark_dicts = read_landmarks(self.landmarks_fpath, self.frame_w, self.frame_h)
        
        #if not self.validation:
        #    print (self.fpath, self.frame_w, self.frame_h, self.frame_cnt)
        self.last_data_fetch = time.time()

    def on_epoch_end(self):
        self.do_next_video()

        self.init_for_file()

    def __get_input(self, idx):

        ret = self.videos[idx][1].asnumpy()[self.current_frame_y][1]
        self.cur_frame[idx] = ret

        buf = self.videos[idx][0].asnumpy()[self.current_frame_y]
        buf_singlechannel = buf[:,:,0:1]

        return buf_singlechannel
    
    def __get_output(self, idx):

        row = self.eli_dicts[idx][self.cur_frame[idx]]
        out = [row["CENTER_X"], row["CENTER_Y"], row["CENTER_X_VEL"], row["CENTER_Y_VEL"]]
        
        return out
    
    def __get_data(self):
        # Generates data containing batch_size samples

        X_batch = []
        y_batch = []
        for i in range(0, self.batch_size):
            X_batch += [self.__get_input(i)]
            y_batch += [self.__get_output(i)]
        

        self.current_frame_y += 1
        if self.current_frame_y >= self.frame_cnt:
            self.current_frame_y = 0

        if not self.validation and int(time.time() - self.last_data_fetch) > 30:
            print ("Taking a long time?")
            #raise TakingTooLongException
            sys.exit(69)
        self.last_data_fetch = time.time()

        return np.asarray(X_batch), np.asarray(y_batch)
    
    def __getitem__(self, index):
        
        X, y = self.__get_data()
        return X, y
    
    def __len__(self):
        return self.frame_cnt