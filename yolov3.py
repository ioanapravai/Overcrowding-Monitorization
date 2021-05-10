import cv2
import numpy as np
import datetime
from threading import Thread
import threading
from queue import Queue
import time
from flask import Flask, render_template, Response

app = Flask(__name__)
global fvs

@app.route("/")
def index():
    # return rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        i = 0
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                i += 1
                if not grabbed:
                    self.stop()
                    return
                if i == 24:
                    self.Q.put(frame)
                    i = 0

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True


def load_yolo():
    # Load YOLO
    # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load video
    # cv2.namedWindow("preview")
    # vc = cv2.VideoCapture('rtsp://administrator:123456@192.168.1.126:554/stream1')
    # vc = cv2.VideoCapture('http://192.168.1.118:554/stream2?loginuse=administrator&loginpas=123456')
    # vc = cv2.VideoCapture(0)
    # Prepare variables for processed frames and dimension
    weight, height = None, None

    # create list with labels
    labels = []
    with open("coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print(labels)
    # Random colors for detected objects
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    return net, layer_names, colors, labels


def start_video():
    fvs = FileVideoStream('rtsp://administrator:123456@192.168.1.126:554/stream1').start()
    time.sleep(1.0)
    return fvs




# catch frames
def generate():
    # global fvs
    fvs = start_video()
    net, layer_names, colors, labels = load_yolo()
    while True:
        frame = fvs.read()


        # if weight is None or height is None:
        height, weight = frame.shape[:2]

        # process frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)


        # YOLO forward step
        net.setInput(blob)

        # get associated probability
        network_output = net.forward(layer_names)

        # prepare lists
        bounding_boxes = []
        confidences = []
        classes = []

        for out in network_output:
            for detected_obj in out:
                scores = detected_obj[5:]
                current_class = np.argmax(scores)
                current_confidence = scores[current_class]

                if current_confidence > 0.5:
                    current_box = detected_obj[0:4] * np.array([weight, height, weight, height])
                    x_center, y_center, box_w, box_h = current_box
                    x_min = int(x_center - (box_w/2))
                    y_min = int(y_center - (box_h/2))

                    # Append results
                    bounding_boxes.append([x_min, y_min, int(box_w), int(box_h)])
                    confidences.append(float(current_confidence))
                    classes.append(current_class)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.)

        # At least one detection should exist
        if len(results) > 0:
            for i in results.flatten():
                # get box coordinates
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_w, box_h = bounding_boxes[i][2], bounding_boxes[i][3]

                # get a color
                color_current_box = colors[classes[i]].tolist()

                # draw box
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_w, y_min + box_h), color_current_box, 2)

                # text label
                current_text_box = '{}: {:.4f}'.format(labels[int(classes[i])], confidences[i])
                cv2.putText(frame, current_text_box, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, color_current_box, 2)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        # cv2.imshow("preview", frame)
        # key = cv2.waitKey(20)
        # if key == 27: # exit on ESC
        #     fvs.stop()
        #     break


# while rval:
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#     # detect_objects(frame)
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#         break
# cv2.destroyWindow("preview")

# # catch frames
# while True:
#     rval, frame = vc.read()
#     if not rval:
#         break
#
#
#
#     if weight is None or height is None:
#         height, weight = frame.shape[:2]
#
#     # process frame
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#
#
#     # YOLO forward step
#     net.setInput(blob)
#
#     # get associated probability
#     network_output = net.forward(layer_names)
#
#     # prepare lists
#     bounding_boxes = []
#     confidences = []
#     classes = []
#
#     for out in network_output:
#         for detected_obj in out:
#             scores = detected_obj[5:]
#             current_class = np.argmax(scores)
#             current_confidence = scores[current_class]
#
#             if current_confidence > 0.5:
#                 current_box = detected_obj[0:4] * np.array([weight, height, weight, height])
#                 x_center, y_center, box_w, box_h = current_box
#                 x_min = int(x_center - (box_w/2))
#                 y_min = int(y_center - (box_h/2))
#
#                 # Append results
#                 bounding_boxes.append([x_min, y_min, int(box_w), int(box_h)])
#                 confidences.append(float(current_confidence))
#                 classes.append(current_class)
#
#     results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.)
#
#     # At least one detection should exist
#     if len(results) > 0:
#         for i in results.flatten():
#             # get box coordinates
#             x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
#             box_w, box_h = bounding_boxes[i][2], bounding_boxes[i][3]
#
#             # get a color
#             color_current_box = colors[classes[i]].tolist()
#
#             # draw box
#             cv2.rectangle(frame, (x_min, y_min), (x_min + box_w, y_min + box_h), color_current_box, 2)
#
#             # text label
#             current_text_box = '{}: {:.4f}'.format(labels[int(classes[i])], confidences[i])
#             cv2.putText(frame, current_text_box, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, color_current_box, 2)
#
#
#     cv2.imshow("preview", frame)
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#         break
#
# if __name__ == 'main':
t = threading.Thread(target=start_video)
t.daemon = True
t.start()
app.run(host="127.0.0.1", port=8000, debug=True, threaded=True, use_reloader=False)

# fvs.stop()
