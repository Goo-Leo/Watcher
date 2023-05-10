import face_recognition
import pickle
import cv2


class FaceRecognizer:
    def __init__(self, encodings_path, detection_method='cnn'):
        self.data = pickle.loads(open(encodings_path, "rb").read())
        self.detection_method = detection_method
        self.face_cascade = cv2.CascadeClassifier('./Signup/haarcascade_frontalface_default.xml')

    def recognize(self):
        cap = cv2.VideoCapture("/dev/video-camera0")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    # 找到所有匹配人脸的索引，然后初始化一个字典来计算每个人脸被匹配的总次数
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # 遍历匹配的索引并为每个识别出的人脸维护一个计数
                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # def updatestatus(self):
    # TODO: update user state in local database


if __name__ == '__main__':
    Login = FaceRecognizer(encodings_path='/home/orangepi/PycharmProjects/Watcher/Signup/encodings.pickle')
    Login.recognize()
