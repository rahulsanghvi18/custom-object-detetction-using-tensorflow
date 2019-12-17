import numpy as np
import tensorflow as tf
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

#load your model in the worling directory and edit the name of the file
#-----------------------------------------------------------------------------------------------------------------------
model_name = 'inference_graph'
model_path = pathlib.Path.cwd()/model_name/'saved_model/'
model = tf.saved_model.load(str(model_path))
model = model.signatures['serving_default']
#modify the test image directory here
#-----------------------------------------------------------------------------------------------------------------------
PATH_TO_LABELS = str(pathlib.Path.cwd()/'training/labelmap.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#PATH_TO_TEST_IMAGES_DIR = pathlib.Path.cwd()/'test_images'
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#-----------------------------------------------------------------------------------------------------------------------

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis, ...]
  output_dict = model(input_tensor)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key: value[0, :num_detections].numpy()
                 for key, value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  if 'detection_masks' in output_dict:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
      output_dict['detection_masks'], output_dict['detection_boxes'],
      image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
  return output_dict



def show_inference(model, image_np):
  output_dict = run_inference_for_single_image(model, image_np)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks = output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates = True,
      line_thickness=8)

  cv.imshow("output window",image_np)

video = cv.VideoCapture(0)

if not video.isOpened():
    print("video canoot be opened")

while True:
    ret,frame = video.read()
    show_inference(model,frame)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()