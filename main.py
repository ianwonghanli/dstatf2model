import cv2
#Loading the saved_model
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
import time
import numpy as np
import warnings
import cv2
warnings.filterwarnings('ignore')
from PIL import Image
# from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
print(tf.__version__)
output = []



PATH_TO_SAVED_MODEL = r"C:\Users\user\Documents\Tensorflow\workspace\training_demo\custom_model\saved_model"
print('Loading model...', end='')

# Load saved model and build the detection function
# tf.keras.models.save_model('saved_model.pb', PATH_TO_SAVED_MODEL)
# detect_fn = tf.keras.models.load_model(PATH_TO_SAVED_MODEL)
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

# Loading the label_map
category_index = label_map_util.create_category_index_from_labelmap(
    r"C:\Users\user\Documents\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt", use_display_name=True)


# category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


# print('Running inference for {}... '.format(image_path), end='')

# image_np = load_image_into_numpy_array(image_path)


def run_inference_and_extract(model, category_index, threshold, label_to_look_for,
                              output_dir, image_path):
    # create output dir if not already created
    output_dir = "C:/Users/user/Documents/Tensorflow/workspace/training_demo/custom_model/saved_model"
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_dir,'/images', exist_ok=True)

    if os.path.exists(output_dir + '/results.csv'):
        df = pd.read_csv(output_dir + '/results.csv')
    else:
        df = pd.DataFrame(columns=['timestamp', 'image_path'])

    image_np = load_image_into_numpy_array(image_path)

    image_show = np.copy(image_np)

    image_height, image_width, _ = image_np.shape

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    print(input_tensor)

    #     # #Actual detection
    #     # output_dict = run_inference_for_single_image(model,image_np)
    #     model = model.signatures["serving_default"]
    detections = detect_fn.predict(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    output_directory = [category_index[1]['name'], ]

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections,
    #     detections['detection_boxes'],
    #     detections['detection_classes'],
    #     detections['detection_scores'],
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=200,
    #     min_score_thresh=.5,  # Adjust this value to set the minimum probability boxes to be classified as True
    #     agnostic_mode=False)

    # cv2.imshow('object_detection', cv2.resize(image_np, (600,600)))

    # get data(label, xmin, ymin, xmax, ymax)

    for index, score in enumerate(detections['detection_scores']):
        if score < 0.5:
            continue
        label = category_index[detections['detection_classes'][index]]['name']
        ymin, xmin, ymax, xmax = detections['detection_boxes'][index]
        print(ymin)
        print(xmax)
        output.append((label, int(xmin * image_width), int(ymin * image_height),
                       int(xmax * image_width), int(ymax * image_height)))

    for label, x_min, y_min, x_max, y_max in output:
        if label == "green digital display":
            array = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
            image = Image.fromarray(array)
            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            file_path = output_dir + '/images/' + str(len(df)) + '.jpg'
            cropped_img.save(file_path, 'JPEG', icc_profile=cropped_img.info.get('icc_profile'))
            df.loc[len(df)] = [datetime.datetime.now(), file_path]
            df.to_csv(output_dir + '/results.csv', index=None)


run_inference_and_extract(detect_fn, category_index, 0.5, "green digital display",
                          "C:/Users/user/Documents/Tensorflow/workspace/training_demo/custom_model/saved_model",
                          'C:/Users/user/Documents/Tensorflow/workspace/training_demo/images/test' + "/F16 short flight_Moment51.jpg")


# # save incident
# for l, x_min, y_min, x_max, y_max in output:
#     if l == "green digital display":
#         array = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
#         image = Image.fromarray(array)
#         cropped_img = image.crop((x_min, y_min, x_max, y_max))
#         file_path = output_dir+'/images/'+str(len(df))+'.jpg'
#         cropped_img.save(file_path, 'JPEG', icc_profile=cropped_img.info.get('icc_profile'))
#         df.loc[len(df)] = [datetime.datetime.now(), file_path]
#         df.to_csv(output_dir+'/results.csv', index=None)