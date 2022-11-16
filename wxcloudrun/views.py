from datetime import datetime
from flask import render_template, request
from run import app
from wxcloudrun.dao import delete_counterbyid, query_counterbyid, insert_counter, update_counterbyid
from wxcloudrun.model import Counters
from wxcloudrun.response import make_succ_empty_response, make_succ_response, make_err_response

import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '欢迎使用微信云托管！'

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 80)))
class PoseDetector():
    '''
    人体姿势检测类
    '''
    def __init__(self,
                 static_image_mode=False,
                 modelComplexity=1,
                 upper_body_only=True,
                 smooth_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        '''
        初始化
        :param static_image_mode: 是否是静态图片，默认为否
        :param upper_body_only: 是否是上半身，默认为否
        :param smooth_landmarks: 设置为True减少抖动
        :param min_detection_confidence:人员检测模型的最小置信度值，默认为0.5
        :param min_tracking_confidence:姿势可信标记的最小置信度值，默认为0.5
        '''
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.modelComplexity=modelComplexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # 创建一个Pose对象用于检测人体姿势
        self.pose = mp.solutions.pose.Pose(
                                        self.static_image_mode,
                                        self.modelComplexity, 
                                        self.upper_body_only, 
                                        self.smooth_landmarks,
                                        self.min_detection_confidence, 
                                        self.min_tracking_confidence)

    def find_pose(self, img, draw=False):
        '''
        检测姿势方法
        :param img: 一帧图像
        :param draw: 是否画出人体姿势节点和连接图
        :return: 处理过的图像
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pose.process(imgRGB) 会识别这帧图片中的人体姿势数据，保存到self.results中
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks,
                                                          mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def find_positions(self, img):
        '''
        获取人体姿势数据
        :param img: 一帧图像
        :param draw: 是否画出人体姿势节点和连接图
        :return: 人体姿势数据列表
        '''
        # 人体姿势数据列表，每个成员由3个数字组成：id, x, y
        # id代表人体的某个关节点，x和y代表坐标位置数据
        self.lmslist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmslist.append([id, cx, cy])

        return self.lmslist

    def find_angle(self, img, p1, p2, p3, draw=False):
        '''
        获取人体姿势中3个点p1-p2-p3的角度
        :param img: 一帧图像
        :param p1: 第1个点
        :param p2: 第2个点
        :param p3: 第3个点
        :param draw: 是否画出3个点的连接图
        :return: 角度
        '''
        x1, y1 = self.lmslist[p1][1], self.lmslist[p1][2]
        x2, y2 = self.lmslist[p2][1], self.lmslist[p2][2]
        x3, y3 = self.lmslist[p3][1], self.lmslist[p3][2]

        # 使用三角函数公式获取3个点p1-p2-p3，以p2为角的角度值，0-180度之间
        angle = int(math.degrees(math.atan2(y1-y2, x1-x2) - math.atan2(y3-y2, x3-x2)))
        if angle < 0:
            angle = angle + 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.circle(img, (x1, y1), 8, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255, 3))
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255, 3))
            cv2.putText(img, str(angle), (x2-50, y2+50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

        return angle


# opencv打开一个视频
def action(point1=24,point2=26,point3=28,angle1=60,angle2=170):

    cap = cv2.VideoCapture("opening.mp4")
    

    # 创建一个PoseDetector类的对象
    detector = PoseDetector()
    # 方向和完成次数的变量
    dir = 0
    count = 0
    start_time=time.time()
    while True:
        # 读取视频图片帧
        success,  img= cap.read()
        img=cv2.resize(img,(1200,800))
        if success:
            # 检测视频图片帧中人体姿势
            img = detector.find_pose(img, draw=False)
            
            # 获取人体姿势列表数据
            lmslist = detector.find_positions(img)


            # 右手肘的角度
            right_angle = detector.find_angle(img, point1, point2, point3,True)
            # 以170到20度检测右手肘弯曲的程度
            right_per = np.interp(right_angle, (angle1, angle2), (100, 0))
            # 进度条高度数据
            right_bar = np.interp(right_angle, (angle1, angle2), (200, 400))
            # 使用opencv画进度条和写右手肘弯曲的程度
            
            # 左手肘的角度
            left_angle = detector.find_angle(img, point1-1, point2-1, point3-1,True)
            left_per = np.interp(left_angle, (angle1, angle2), (100, 0))
            left_bar = np.interp(left_angle, (angle1, angle2), (200, 400))
           
            cv2.rectangle(img, (1000, 200), (1020, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (1000, int((right_bar+left_bar)/2)), (1020, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int((right_per+left_per)/2)) + '%', (980, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            head_angle=detector.find_angle(img,28,0,27,True)
            head_per=np.interp(head_angle, (0, 25), (0, 100))
            head_bar=np.interp(head_angle*6, (angle1,angle2 ), (200, 400))
            cv2.rectangle(img, (200, 200), (220, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (200, int(head_bar)), (220, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(head_per)) + '%', (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
           

            # 检测个数，我这里设置的是从20%做到80%，就认为是一个
            if (left_per >= 70 and right_per >= 70):
                if dir == 0:
                    count = count + 1.0/3.0
                    dir = 1
            if (left_per <= 40 and right_per <= 40):
                if dir == 1:
                    count = count + 1.0/3.0
                    dir = 2
            if ( head_angle<= 20):

                if dir == 2:
                    count = count + 1.1/3.0
                    dir = 0
            now = time.time()
            fps_time = now - start_time
            start_time = now
            fps_txt =round( 1/fps_time,2)
            cv2.putText(img, str(fps_txt), (50,200), cv2.FONT_ITALIC, 1, (0,255,0),2)
            # 在视频上显示完成个数
            cv2.putText(img, str(int(count)), (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
            
            cv2.imshow('Image',img)
        else:
            break
        k = cv2.waitKey(1)
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__=="__main__":
    action(point1=14,point2=12,point3=24,angle1=10,angle2=170)
cv2.release()
cv2.destroyAllWindows()
