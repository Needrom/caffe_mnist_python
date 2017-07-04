# -*- coding: utf-8 -*-
import cv2
import caffe
import numpy as np
import wx
from contours_test import contours_test
from grayify_test import grapify_test
from corrosion_expand import corrsion_expand



caffe_root = 'F:/caffe-windows/' #路径

MODEL_FILE = caffe_root + 'examples/mnist/lenet.prototxt'
PRETRAINED = caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
caffe.set_mode_gpu()     #使用GPU模式

IMAGE_PATH = 'F:/create_mnist_data2/train/'    #图片路径
font = cv2.FONT_HERSHEY_SIMPLEX

def resize(image):
    image_path = "./screenshot.bmp"
    dst = cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)   #改变图片大小
    return dst

def check_test(image):

    input_image = image.astype(np.float32)
    resized = cv2.resize(input_image, (280, 280), None, 0, 0, cv2.INTER_AREA)
    input_image = input_image[:, :, np.newaxis]

    prediction = net.predict([input_image], oversample=False)  #预测函数
    cv2.putText(resized, str(prediction[0].argmax()), (200, 280), font, 4, (255,), 2, cv2.LINE_AA)
    #保存预测后的图片
    #cv2.imwrite('F:\\pic\\result.jpg',resized)
    #cv2.imshow("Prediction", resized)
    print 'predicted class:', prediction[0].argmax()

    return prediction[0].argmax()
    #frame1.myPic3=wx.StaticBitmap(frame1.panel, -1, pos=(500, 100))
    #pic3 = wx.Bitmap("F:\pic\pig.jpg", wx.BITMAP_TYPE_ANY)
    #frame1.myPic3.SetBitmap(pic3)

    # astype() is amethod provided by numpy to convert numpy dtype.
    # resize Image to improve vision effect.
    # input_image.shape is (28, 28, 1), with dtype float32
    # The previous two lines(exclude resized line) is the same as what caffe.io.load_iamge() do.
    # According to the source code, caffe load_image uses skiamge library to load image from disk.

    # for debug
    # print type(input_image), input_image.shape, input_image.dtype
    # print input_image

def show_box(frame):
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 48, 255, cv2.THRESH_BINARY_INV)
    thresh = corrsion_expand(thresh)
    #cv2.imshow("corrsion",thresh)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # for i in img_list:
        # cv2.imwrite("F:\\pic\\img_list",i)
        # cv2.waitKey(0)


    # 用红色表示有旋转角度的矩形框架
    for cnt in contours[1]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imshow("rect", img)
    count = 0

class MyFrame(wx.Frame):

    def __init__(self, parent,id):
        wx.Frame.__init__(self, parent, id, u'手写字体识别', size=(1000, 800))
        #定义面板
        self.panel = wx.Panel(self)
        self.key = 0
        self.img_list = []
        self.res_list = []
        #设置背景图片
        #backgroundImage_file = 'F:\\pic\\bg.jpg'
        #to_bmp_image = wx.Image(backgroundImage_file, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        #self.bitmap = wx.StaticBitmap(self.panel, -1, to_bmp_image, (0, 0))

        #设置各类图片
        #图片1
        #self.myPic1 = wx.StaticBitmap(self.panel, -1, pos=(100, 100))
        #pic1 = wx.Bitmap("F:\\pic\\res.jpg", wx.BITMAP_TYPE_ANY)
        #self.myPic1.SetBitmap(pic1)
        #图片2
        #self.myPic2 = wx.StaticBitmap(self.panel, -1, pos=(500, 100))
        #pic2 = wx.Bitmap("F:\pic\pig.jpg", wx.BITMAP_TYPE_ANY)
        #self.myPic2.SetBitmap(pic2)

        # 轮廓图片
        # self.myPic1 = wx.StaticBitmap(self.panel, -1, pos=(100, 100))
        # pic1 = wx.Bitmap("F:\\pic\\resStart.jpg", wx.BITMAP_TYPE_ANY)
        # self.myPic1.SetBitmap(pic1)
        # 预测结果图片
        # self.myPic2 = wx.StaticBitmap(self.panel, -1, pos=(400, 100))
        # pic2 = wx.Bitmap("F:\\pic\\resultStart.jpg", wx.BITMAP_TYPE_ANY)
        # self.myPic2.SetBitmap(pic2)

        # str = "This is a different font."
        # text = StaticText(self, -1, str, (20, 120))
        # font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
        # text.SetFont(font)

        #添加静态文本
        str1 = u"检测到的数值："
        str2 = u"识别到的值  ："
        text1 = wx.StaticText(self.panel, -1, str1, (60, 150),)
        text2 = wx.StaticText(self.panel, -1, str2, (60, 400))
        font1 = wx.Font(20, wx.SWISS, wx.NORMAL, wx.NORMAL)
        font2 = wx.Font(20, wx.SWISS, wx.NORMAL, wx.NORMAL)
        text1.SetFont(font1)
        text2.SetFont(font2)
        # wx.StaticText(self.panel, -1, u"检测到的数值：",(80, 150),size=(100,100))
        # wx.StaticText(self.panel, -1, u"识别到的值  ：",(80, 400))

        # 图片更新
        # self.bmp = wx.BitmapFromBuffer(28, 28, None)
        # self.displayPanel.Bind(wx.EVT_PAINT, self.onPaint)


        #创建按钮
        button_close = wx.Button(self.panel, label=u'关闭', pos=(10, 10), size=(100, 30))
        button_open_camera = wx.Button(self.panel, label=u'打开摄像头', pos=(120, 10), size=(100, 30))
        button_pre=wx.Button(self.panel, label=u'识别', pos=(240, 10), size=(100, 30))
        button_takephoto = wx.Button(self.panel, label=u'拍照', pos=(360, 10), size=(100, 30))
        # 绑定单击事件
        self.Bind(wx.EVT_BUTTON, self.OnCloseMe, button_close)
        self.Bind(wx.EVT_BUTTON, self.OpenCamera, button_open_camera)
        self.Bind(wx.EVT_BUTTON, self.PreResult, button_pre)
        self.Bind(wx.EVT_BUTTON, self.TakePhoto, button_takephoto)
    #按钮单击事件
    def OnCloseMe(self, event):
        self.key = 2
        self.Close(True)

    def OpenCamera(self, event):
        #self.res_list = []
        capture = cv2.VideoCapture(0)
        grap = grapify_test()
        _, frame = capture.read()
        while frame is not None:
            #cv2.imshow('origen', frame)  # 在窗口中显示图片
            show_box(frame)
            frame = grap.grapify_thresholding(frame)
            cv2.waitKey(10)

            if self.key == 1:
                cv2.imwrite('screenshot.bmp', frame)
                self.img_list = contours_test("screenshot.bmp")
                if self.img_list != None:
                    for i in range(0,self.img_list.__len__()):
                        input_image = cv2.imread('resize_{}.bmp'.format(i), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        self.res_list.append(check_test(input_image))
                self.key = 0

            elif self.key == 2:
                break
            _, frame = capture.read()

    def TakePhoto(self, event):
        self.res_list = []
        self.key = 1
    def PreResult(self, event):
        # 轮廓图片
        for i in range(0, self.img_list.__len__()):
            #input_image = cv2.imread('resize_{}.bmp'.format(i), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            frame.myPic3 = wx.StaticBitmap(frame.panel, -1, pos=(250+100*i, 150))
            pic3 = wx.Bitmap('resize_{}.bmp'.format(i), wx.BITMAP_TYPE_ANY)
            frame.myPic3.SetBitmap(pic3)

        # 预测结果
        for i in range(0, self.res_list.__len__()):

            # str = "This is a different font."
            # text = StaticText(self, -1, str, (20, 120))
            # font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
            # text.SetFont(font)
            print i
            if self.res_list[i] == 0:
                str = "0"
                text = wx.StaticText(self.panel, -1, str, (280 + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 1:
                str = "1"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 2:
                str = "2"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 3:
                str = "3"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 4:
                str = "4"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 5:
                str = "5"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 6:
                str = "6"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 7:
                str = "7"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 8:
                str = "8"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            if self.res_list[i] == 9:
                str = "9"
                text = wx.StaticText(self.panel, -1, str, (280  + 100 * i, 370))
                font = wx.Font(40, wx.SWISS, wx.NORMAL, wx.NORMAL)
                text.SetFont(font)

            #wx.StaticText(self.panel, -1, self.res_list[i], (100+50*i, 200))
            #frame.myPic2 = wx.StaticBitmap(frame.panel, -1, pos=(100+50*i, 250))
            #cv2.imwrite('./123.bmp',self.res_list[i])
            # frame.myPic2 = wx.StaticBitmap(frame.panel, -1, pos=(100 + 50 * i, 250))
            # pic2 = wx.Bitmap('./123.bmp', wx.BITMAP_TYPE_ANY)
            # frame.myPic2.SetBitmap(pic2)

if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyFrame(parent=None, id=-1)
    frame.Show()

    app.MainLoop()