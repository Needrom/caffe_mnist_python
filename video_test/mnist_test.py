import sys
import numpy as np
import cv2
import caffe

caffe_root = 'F:/caffe-windows/'

MODEL_FILE = caffe_root + 'examples/mnist/lenet.prototxt'
PRETRAINED = caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'
CURRENT_DIR = 'G:/create_mnist_data/'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
caffe.set_mode_cpu()

IMAGE_PATH = 'G:/create_mnist_data/train/'
font = cv2.FONT_HERSHEY_SIMPLEX

# for i in range(0, 200):
#     # astype() is a method provided by numpy to convert numpy dtype.
#     input_image = cv2.imread(IMAGE_PATH + 'train_{}.bmp'.format(i), cv2.IMREAD_GRAYSCALE).astype(np.float32)
#     resized = cv2.resize(input_image, (280, 280), None, 0, 0, cv2.INTER_AREA)
#     # resize Image to improve vision effect.
#     input_image = input_image[:, :, np.newaxis] # input_image.shape is (28, 28, 1), with dtype float32
#     # The previous two lines(exclude resized line) is the same as what caffe.io.load_iamge() do.
#     # According to the source code, caffe load_image uses skiamge library to load image from disk.
#
#     # for debug
#     # print type(input_image), input_image.shape, input_image.dtype
#     # print input_image
#
#     prediction = net.predict([input_image], oversample=False)
#     cv2.putText(resized, str(prediction[0].argmax()), (200, 280), font, 4, (255,), 2, cv2.LINE_AA)
#     cv2.imshow("Prediction", resized)
#     print 'predicted class:', prediction[0].argmax()
#     keycode = cv2.waitKey(0) & 0xFF
#     if keycode == 27:
#         break


def check_test(path):
    # astype() is a method provided by numpy to convert numpy dtype.
    input_image = cv2.imread(CURRENT_DIR + path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    resized = cv2.resize(input_image, (280, 280), None, 0, 0, cv2.INTER_AREA)
    # resize Image to improve vision effect.
    input_image = input_image[:, :, np.newaxis] # input_image.shape is (28, 28, 1), with dtype float32
    # The previous two lines(exclude resized line) is the same as what caffe.io.load_iamge() do.
    # According to the source code, caffe load_image uses skiamge library to load image from disk.

    # for debug
    # print type(input_image), input_image.shape, input_image.dtype
    # print input_image

    prediction = net.predict([input_image], oversample=False)
    cv2.putText(resized, str(prediction[0].argmax()), (200, 280), font, 4, (255,), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", resized)
    print 'predicted class:', prediction[0].argmax()
    keycode = cv2.waitKey(0) & 0xFF

if __name__ == "__main__":
    check_test("train/train_0.bmp")