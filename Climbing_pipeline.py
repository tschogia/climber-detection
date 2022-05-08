# Lint as: python2, python3
# This is the work of Gian Tschopp for detecting Climbers in indoor sportsclibing scenes
#
# ==============================================================================

import os  # importing OS in order to make GPU visible
import statistics
import sys  # importyng sys in order to access scripts located in a different folder
from numpy.linalg import norm
import tensorflow as tf  # import tensorflow
import matplotlib
from matplotlib import pyplot as plt

# other import
import numpy as np
from PIL import Image

matplotlib.use('Qt5Agg')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TODO: specify your computational device


# # checking that GPU is found
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# import Scripts already provided by Tensorflow
path2scripts = 'models/research/'  # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts)  # making scripts in models/research available for import

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from keras.models import load_model

# Model config of Trained model
# TODO swich model here, when new model is ready
path2config = 'workspace/exported_models/centernet_1/v3/pipeline.config'
path2model = 'workspace/exported_models/centernet_1/v3/checkpoint'

configs = config_util.get_configs_from_pipeline_file(path2config)  # importing config
model_config = configs['model']  # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False)  # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

path2label_map = 'workspace/data/label_map.pbtxt'  # TODO: provide a path to the label map file
category_index = label_map_util.create_category_index_from_labelmap(path2label_map, use_display_name=True)



###### Functions #######

def detect_fn(image):
    """
    Detect objects in image.

    Args:
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """

    return np.array(Image.open(path))


def inference_with_plot(path2images, box_th=0.25):
    """
    Function that performs inference and plots resulting b-boxes

    Args:
      path2images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction.

    Returns:
      None
    """
    for image_path in path2images:
        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=box_th,
            agnostic_mode=False,
            line_thickness=5)

        plt.figure(figsize=(15, 10))
        plt.imshow(image_np_with_detections)
        print('Done')
    plt.show()


def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


def inference_as_raw_output(path2images, box_th=0.25, nms_th=0.5, to_file=False, data=None, path2dir=False):
    """
   Function that performs inference and return filtered predictions

   Args:
     path2images: an array with pathes to images
     box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
     nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
     to_file: (boolean). When passed as True => results are saved into a file. Writing format is
     path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
     data: (str) name of the dataset you passed in (e.g. test/validation)
     path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.

   Returs:
     detections (dict): filtered predictions that model made
   """


    print(f'Current data set is {data}')
    print(f'Ready to start inference on {len(path2images)} images!')

    for image_path in path2images:

        if path2dir:  # if a path to a directory where images are stored was passed in
            image_path = os.path.join(path2dir, image_path.strip())

        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))

        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}

        if box_th:  # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array

        if nms_th:  # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                   detections['detection_scores'],
                                   detections['detection_classes']
                                   )
                               )
            boxes, scores, classes = nms(output_info)

            detections['detection_boxes'] = boxes  # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes

        if to_file and data:  # if saving to txt file was requested

            image_h, image_w, _ = image_np.shape
            file_name = f'pred_result_{data}.txt'

            line2write = list()
            line2write.append(os.path.basename(image_path))

            with open(file_name, 'a+') as text_file:
                # iterating over boxes
                for b, s, c in zip(boxes, scores, classes):
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w

                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])

                    line2write.append(line2append)

                line2write = ' '.join(line2write)
                text_file.write(line2write + os.linesep)

        return detections


def inference_as_raw_output_new(frame, box_th=0.25, nms_th=0.5):
    """
   Function that performs inference and return filtered predictions

   Args:
     path2images: an array with pathes to images
     box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
     nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
     to_file: (boolean). When passed as True => results are saved into a file. Writing format is
     path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
     data: (str) name of the dataset you passed in (e.g. test/validation)
     path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.

   Returs:
     detections (dict): filtered predictions that model made
   """


    #image_np = load_image_into_numpy_array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    # checking how many detections we got
    num_detections = int(detections.pop('num_detections'))
    # filtering out detection in order to get only the one that are indeed detections
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    # defining what we need from the resulting detection dict that we got from model output
    key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
    # filtering out detection dict in order to get only boxes, classes and scores
    detections = {key: value for key, value in detections.items() if key in key_of_interest}
    if box_th:  # filtering detection if a confidence threshold for boxes was given as a parameter
        for key in key_of_interest:
            scores = detections['detection_scores']
            current_array = detections[key]
            filtered_current_array = current_array[scores > box_th]
            detections[key] = filtered_current_array
    if nms_th:  # filtering rectangles if nms threshold was passed in as a parameter
        # creating a zip object that will contain model output info as
        output_info = list(zip(detections['detection_boxes'],
                               detections['detection_scores'],
                               detections['detection_classes']
                               )
                           )
        boxes, scores, classes = nms(output_info)
        detections['detection_boxes'] = boxes  # format: [y1, x1, y2, x2]
        detections['detection_scores'] = scores
        detections['detection_classes'] = classes

    return detections

def plot_pedestrian_boxes_on_image(pedestrian_boxes):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    thickness = 2
    # color_node = (192, 133, 156)
    color_node = (160, 48, 112)
    # color_10 = (80, 172, 110)
    climbersInFrame = 0
    for i in range(len(pedestrian_boxes)):
        climbersInFrame += 1
        pt1 = (
            int(pedestrian_boxes[i][1] * frame_w),
            int(pedestrian_boxes[i][0] * frame_h),
        )
        pt2 = (
            int(pedestrian_boxes[i][3] * frame_w),
            int(pedestrian_boxes[i][2] * frame_h),
        )

        xpoint = (pt1[0] + pt2[0]) / 2
        ypoint = (pt1[1] + pt2[1]) / 2

        boxHalfWith = pt2[0] - pt1[0]
        boxHalfHight = pt2[1] - pt1[1]

        if len(lstm_input) >= 4:
            del lstm_input[0]
        lstm_input.append([ypoint, xpoint])
        cnnMean_Point = [ypoint.__round__(), xpoint.__round__()]

        print("Personenpunkt:", xpoint, " ", ypoint)
        if (climbersInFrame > 1):
            print("2 Personen oder mehr erkannt: ", climbersInFrame)

        frame_with_boxes = cv2.rectangle(frame, pt1, pt2, color_node, thickness)
        return frame_with_boxes, cnnMean_Point, boxHalfWith, boxHalfHight, pt1, pt2


def prep_frame_for_LSTM(detection_boxes):
    global new_image, frame_h, frame_w, i
    new_image = np.zeros(frame.shape)
    pedestrian_boxes = detection_boxes
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    for i in range(len(pedestrian_boxes)):
        x1 = int(pedestrian_boxes[i][1] * frame_w)
        x2 = int(pedestrian_boxes[i][3] * frame_w)
        y1 = int(pedestrian_boxes[i][0] * frame_h)
        y2 = int(pedestrian_boxes[i][2] * frame_h)
        climber = frame[y1:y2, :][:, x1:x2:, ]
        new_image[y1:y2, :][:, x1:x2:, ] = climber
        new_image = new_image.astype('uint8')
        small = cv2.resize(new_image, (0, 0), fx=0.3, fy=0.3)

    return small

def predict_LSTM_next_point(cnn_frame):
    if (len(lstm_input) < 4):
        print ("nicht genug punkte für prediction")
        return np.array([])
    else:
        noramlized_input = np.array(lstm_input) / 255
        new_prediction = lstm_model.predict(np.expand_dims(noramlized_input, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)[0]
        predicted_frame = np.round(predicted_frame * 255)
        lstm_mean_point = (predicted_frame[1].astype('int'), predicted_frame[0].astype('int'))
        cnn_frame = cv2.circle(cnn_frame, (predicted_frame[1].astype('int'),predicted_frame[0].astype('int')), radius=10, color=(255, 0, 255), thickness=-1)
        print ("prediction LSTM Point: ", predicted_frame)
        return cnn_frame, predicted_frame, lstm_mean_point

def plot_actualbox_to_frame(frame):
    thickness = 2
    # color_node = (192, 133, 156)
    color_node = (0, 0, 255)
    print (actual)
    box = np.array(df.iloc[[actual]]['bbox'].values[0])

    p1 = (box[0].__round__(),
          (box[1]).__round__(),
          )
    p2 = ((box[0] + box[2]).__round__(),
         (box[1] + box[3]).__round__(),
         )
    frame_with_boxes = cv2.rectangle(frame, p1, p2, color_node, thickness)
    return frame_with_boxes, p1, p2

def plot_acutal_mean_pomit_on_frame(meanFrame):
    box = np.array(df.iloc[[actual]]['bbox'].values[0])

    acutal_meanX = ((box[0] + box[0] + box[2]) / 2).__round__()
    acutal_meanY = ((box[1] + box[1] + box[3]) / 2).__round__()
    acutal_mean = (acutal_meanX, acutal_meanY,)

    return cv2.circle(meanFrame, acutal_mean, radius=10, color=(0, 0, 255), thickness=-1), acutal_mean


def plot_mean_point_on_frame(frame12, cnnMeanPoint, lstmPredictedPoint):

    meanPointX = statistics.mean([cnnMeanPoint[1], lstmPredictedPoint[1].__round__()])
    meanPointY = statistics.mean([cnnMeanPoint[0], lstmPredictedPoint[0].__round__()])
    meanFrame = cv2.circle(frame12, (meanPointX.__round__(), meanPointY.__round__()), radius=10,
                           color=(0, 255, 0), thickness=-1)
    return meanFrame

def plot_mean_box_on_frame(frame11, cnnMeanPoint, lstmPredictedPoint, boxW, boxH):
    thickness = 2
    color_node = (0, 255, 0)


    meanPointX = statistics.mean([cnnMeanPoint[1], lstmPredictedPoint[1].__round__()])
    meanPointY = statistics.mean([cnnMeanPoint[0], lstmPredictedPoint[0].__round__()])

    boxp1 = (
        int(meanPointX - (boxW / 2)),
        int(meanPointY + (boxH / 2)),
    )
    boxp2 = (
        int(meanPointX + (boxW / 2)),
        int(meanPointY - (boxH / 2)),
    )

    meanFrame = cv2.rectangle(frame11, boxp1, boxp2, color_node, thickness)
    return meanFrame

def calculate_iou(p1, p2, pt1, pt2, true_positive, false_positive):
    xA = max(p1[0], pt1[0])
    yA = max(p1[1], pt1[1])
    xB = min(p2[0], pt2[0])
    yB = min(p2[1], pt2[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (p2[0] - p1[0] + 1) * (p2[1] - p1[1] + 1)
    boxBArea = (pt2[0] - pt1[0] + 1) * (pt2[1] - pt1[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    if iou >= 0.5:
        true_positive += 1
    else:
        false_positive += 1

    iou_list.append(iou)
    print("IoU Mean: ", np.mean(iou_list))
    print("IoU: ", iou)

    return true_positive, false_positive

def calculate_cos_similarity(a, b):
    ausgangsPunkt = np.array(lstm_input[-1])
    zwischenspeicher = ausgangsPunkt[0]
    ausgangsPunkt[0] = ausgangsPunkt[1]
    ausgangsPunkt[1] = zwischenspeicher
    a_rel = a - ausgangsPunkt
    b_rel = b - ausgangsPunkt
    cos_sim = a_rel@b_rel / (norm(a_rel) * norm(b_rel))

    cosSimList.append(abs(cos_sim))
    print("Cos Sim: ", np.mean(cosSimList))


    data = cosSimList
    fig = plt.figure(figsize=(10, 7))
    # Creating plot
    plt.boxplot(data)
    # show plot
    plt.show()

    return cos_sim


# def preict_lstm_next_frame():
#     if len(lstm_prediction_frames) > 4:
#         for frame in lstm_prediction_frames:
#             cv2.namedWindow("input", cv2.WINDOW_NORMAL)
#             cv2.imshow("input", frame)
#             cv2.waitKey(1)
#
#         prediction_frames = np.concatenate(lstm_prediction_frames).reshape(-1, 324, 576, 3)
#         prediction_frames = prediction_frames / 255
#         new_prediction = lstm_model.predict(np.expand_dims(prediction_frames, axis=0))
#         new_prediction = np.squeeze(new_prediction, axis=0)
#         predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)[0]
#         return predicted_frame
#     else:
#         return np.array([])


# def add_frames_to_lstm_prediction_set(lstm_frame):
#     if len(lstm_prediction_frames) > 4:
#         del lstm_prediction_frames[0]
#     lstm_prediction_frames.append(lstm_frame)


##### Main ######
import cv2

cap = cv2.VideoCapture('winti_sie_test.m4v')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = cap.get(cv2.CAP_PROP_FPS)
print (fps)

frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1
actual = 0
iou_list = []
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
cosSimList = []

##### test anotations ######

import json
import pandas as pd

# Opening JSON file
f = open('instances_default.json')

# returns JSON object as
# a dictionary
data = json.load(f)
# Closing file
f.close()
df = pd.DataFrame(data['annotations'])

newDf = df[:3000]


# lstm_model = load_model('lstm_model.h5')
# lstm_prediction_frames = []

lstm_model = load_model('lstm_point.h5')
lstm_model.summary()

lstm_input = []

while cap.isOpened():
    frame_num += 1
    frame_count = 1
    #actual += 1
    while (frame_count < 15):
        ret, frame = cap.read()
        frame_count += 1
        actual += 1
    #ret, frame = cap.read()

    if not ret:
        #lstm_input = np.array(lstm_input)
        #np.save('lstm_imput3.npy', lstm_input)
        print("end of the video file...")
        break

    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    original_frame = frame.copy()

    print("Processing frame: ", frame_num)

    detections = inference_as_raw_output_new(frame)
    print(detections['detection_boxes'])

    cv2.startWindowThread()

    if len(detections['detection_boxes']) > 0:
        # set bounding-boxes around climbers
        frameWithBoxes, cnnMeanPoint, box_with, box_hight, pt1, pt2 = plot_pedestrian_boxes_on_image(detections['detection_boxes'])
        frameWithBoxes, p1, p2 = plot_actualbox_to_frame(frameWithBoxes)
        true_positive, false_positive = calculate_iou(p1, p2, pt1, pt2, true_positive, false_positive)

        # set black background for lstm Model prediction
        # add_frames_to_lstm_prediction_set(lstm_frames)

        cv2.namedWindow("Kletterhalle", cv2.WINDOW_NORMAL)
        cv2.imshow("Kletterhalle", frameWithBoxes)
        # cv2.imwrite("workspace/lstm_imput/winti1_small4/frame%d.jpg" % frame_num, lstm_frames)
        cv2.waitKey(1)

        # predicted_frame = predict_LSTM_next_point(frameWithBoxes)
        if len(lstm_input) > 3:
            predicted_frame, lstmPredictedPoint, lstm_mean_point = predict_LSTM_next_point(original_frame)

            if len(predicted_frame) > 0:
                ### Plot combination point
                predicted_frame = plot_mean_point_on_frame(predicted_frame, cnnMeanPoint, lstmPredictedPoint)

                ### Plot actual mean point from annotations to mean frame
                # plotframe = plot_acutal_mean_pomit_on_frame(meanFrame)
                ### Plot actual mean point from annotations to LSTM frame
                plotframe, acutual_mean_point = plot_acutal_mean_pomit_on_frame(predicted_frame)

                ### calculate cosinus similarity
                cos_sim = calculate_cos_similarity(acutual_mean_point, lstm_mean_point)
                print ("cos-sim:", cos_sim)

                ### Combination box when box exists
                #meanFrame = plot_mean_box_on_frame(original_frame, cnnMeanPoint, lstmPredictedPoint, box_with, box_hight)

                ### Plot original Box from annotations on the frame
                #original_frame = plot_actualbox_to_frame(original_frame)

                cv2.namedWindow("Kletterhalle-Mean-Point", cv2.WINDOW_NORMAL)
                cv2.imshow("Kletterhalle-Mean-Point", plotframe)
                cv2.waitKey(1)
            else:
                print ("not enough items in list to predict on LSTM")

        lstm_frames = prep_frame_for_LSTM(detections['detection_boxes'])


    else:
        frameWithBoxes, unused, unused2 = plot_actualbox_to_frame(original_frame)
        false_negative += 1
        cv2.namedWindow("Kletterhalle", cv2.WINDOW_NORMAL)
        cv2.imshow("Kletterhalle", frameWithBoxes)
        cv2.waitKey(1)

    print ("TP: ", true_positive, " FP: ", false_positive, " TN: ", true_negative, " FN: ", false_negative)
    if true_positive > 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        print ("Precision: ", precision)
        print ("Recall: ", recall)

    # else:
    #     lstmOnlyOutput = predict_LSTM_next_point(frame)
    #     if len(predicted_frame) > 0:
    #         cv2.namedWindow("Kletterhalle-LSTM", cv2.WINDOW_NORMAL)
    #         cv2.imshow("Kletterhalle-LSTM", lstmOnlyOutput)
    #         cv2.waitKey(1)
    #     else:
    #         print ("not enough items in list to predict on LSTM")
    #         cv2.namedWindow("Kletterhalle-LSTM", cv2.WINDOW_NORMAL)
    #         cv2.imshow("Kletterhalle-LSTM", frame)
    #         cv2.waitKey(1)



    #inference_as_raw_output(["workspace/data/videos/testsetpngs/frame_87.png", "workspace/data/videos/testsetpngs/frame_50.png", "workspace/data/videos/testsetpngs/frame_10.png", "workspace/data/videos/testsetpngs/frame_20.png", "wintipics/frame-1029.png"])
