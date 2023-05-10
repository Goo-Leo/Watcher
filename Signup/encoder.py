import pickle
from imutils import paths
import face_recognition
import cv2
import os

dataset_path = 'Faces'
encodings_path = '../encodings.pickle'
detection_method = 'cnn'


class Face_Encoder():
    def __init__(self, dataset_path='Faces', encodings_path='encodings.pickle', detection_method='cnn'):
        self.dataset_path = dataset_path
        self.encodings_path = encodings_path
        self.detection_method = detection_method

    def encode_images(self):
        # 获取数据集中输入图像的路径
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.dataset_path))
        # 初始化已知编码和已知名称的列表
        knownEncodings = []
        knownNames = []
        # 遍历图像路径
        for (i, imagePath) in enumerate(imagePaths):
            # 从图片路径中提取人名
            print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))


