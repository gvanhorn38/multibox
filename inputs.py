"""
Feed images and their bounding boxes into the network. 
"""

import numpy as np
from scipy.misc import imresize
import tensorflow as tf
import sys 

def resize_image_maintain_aspect_ratio(image, target_height, target_width):
  """
  image needs to be RGB image that has been casted to float32
  """
  
  height, width, _ = image.shape
  
  if height > width:
    new_height = target_height
    height_factor = float(1.0)
    width_factor = new_height / float(height)
    new_width = int(np.round(width * width_factor))
  else:
    new_width = target_width
    width_factor = float(1.0)
    height_factor = new_width / float(width)
    new_height = int(np.round(height * height_factor))
  
  resized_image = imresize(
    image,
    (new_height, new_width)
  )
  
  output = np.pad(resized_image, ((0, target_height - new_height), (0, target_width - new_width), (0, 0)), 'constant', constant_values=0).astype(np.float32)
  
  return [output, np.int32(new_height), np.int32(new_width)] 

def augment_image_and_bboxes(image, orig_bboxes, do_random_flip, do_random_shift, max_shift, do_random_crop):
  """
  Perturb the bounding boxes in the image.
  image : np.array
  orig_bboxes: np.array [[x1, y1, x2, y2]] Normalized coordinates
  """
  
  image_height, image_width = image.shape[:2]
  
  # Sanity check the bounding boxes
  perturbed_bboxes = []
  for bbox in orig_bboxes:
    x1, y1, x2, y2 = bbox
    if x1 > x2:
      t = x1
      x1 = x2
      x2 = t
    if y1 > y2:
      t = y1
      y1 = y2
      y2 = t
    
    x1 = max(0., x1)
    y1 = max(0., y1)
    x2 = min(1., x2)
    y2 = min(1., y2)
    
    perturbed_bboxes.append([x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height])
  
  perturbed_bboxes = np.array(perturbed_bboxes, dtype=np.float32)
  perturbed_image = image
  
  # Randomly flip the image
  if do_random_flip and np.random.rand() < 0.5:
    perturbed_image = np.fliplr(perturbed_image)
    flipped_bboxes = []
    for bbox in perturbed_bboxes:
      x1, y1, x2, y2 = bbox
      flipped_bboxes.append([image_width - x2, y1, image_width - x1, y2])
    
    perturbed_bboxes = np.array(flipped_bboxes)
  
  if do_random_shift:
    max_radius_of_wiggle = max_shift # 10
    shifted_bboxes = []
    for bbox in perturbed_bboxes:
      
      x1, y1, x2, y2 = bbox
      
      # Determine the valid coordinates for the perturbed top left bbox corner
      # The first value specifies the max "backward" coordinate
      # The second value specifies the max "forward" coordinate
      top_left_x_valid_positions = [max(0, x1 - max_radius_of_wiggle),
                                    min(x1 + max_radius_of_wiggle, image_width)]
      top_left_y_valid_positions = [max(0, y1 - max_radius_of_wiggle),
                                    min(y1 + max_radius_of_wiggle, image_height)]
      # Randomly choose a new point                          
      perturbed_x1 = np.random.randint(top_left_x_valid_positions[0], top_left_x_valid_positions[1]+1)
      perturbed_y1 = np.random.randint(top_left_y_valid_positions[0], top_left_y_valid_positions[1]+1)

      # Determine the valid coordinates for the perturbed bottom right bbox corner
      # The first value specifies the max "backward" coordinate
      # The second value specifies the max "forward" coordinate
      bottom_right_x_valid_positions = [max(0, x2 - max_radius_of_wiggle),
                                        min(x2 + max_radius_of_wiggle, image_width)]
      bottom_right_y_valid_positions = [max(0, y2 - max_radius_of_wiggle),
                                        min(y2 + max_radius_of_wiggle, image_height)]
      # Randomly choose a new point
      perturbed_x2 = np.random.randint(bottom_right_x_valid_positions[0], bottom_right_x_valid_positions[1]+1)
      perturbed_y2 = np.random.randint(bottom_right_y_valid_positions[0], bottom_right_y_valid_positions[1]+1)
      
      shifted_bboxes.append([perturbed_x1, perturbed_y1, perturbed_x2, perturbed_y2])
    
    perturbed_bboxes = np.array(shifted_bboxes)
    
  if do_random_crop:
    """
    Get the distance from each edge of the image to the nearest bbox edge. Then randomly crop the image.
    """  
    
    min_x1 = np.min(perturbed_bboxes[:, 0])
    min_y1 = np.min(perturbed_bboxes[:, 1])
    max_x2 = np.max(perturbed_bboxes[:, 2])
    max_y2 = np.max(perturbed_bboxes[:, 3])
    
    left = np.random.randint(0, min_x1 + 1)
    top = np.random.randint(0, min_y1 + 1)
    right = np.random.randint(max_x2, image_width + 1)
    bottom = np.random.randint(max_y2, image_height + 1)
    
    perturbed_image = perturbed_image[top:bottom, left:right, :]
    perturbed_bboxes = perturbed_bboxes - np.array([left, top, left, top])
    
  
  image_height, image_width = perturbed_image.shape[:2]
  
  image_height = float(image_height)
  image_width = float(image_width)
  
  # Do a sanity check on the bbox coordinates, and normalize them 
  gtg_bboxes = []
  for bbox in perturbed_bboxes:
    x1, y1, x2, y2 = bbox
    gtg_bboxes.append([
      max(0, x1 / image_width),
      max(0, y1 / image_height),
      min(1, x2 / image_width),
      min(1, y2 / image_height)
    ])
  perturbed_bboxes = np.array(gtg_bboxes, dtype=np.float32)
  
  return [perturbed_image, perturbed_bboxes, np.int32(image_height), np.int32(image_width)]


def reshape_bboxes(bboxes):
  """ Utility function to help reshape bounding boxes for rendering through tensorboard. 
  Convert bboxes from [x1, y1, x2, y2] to [y1, x1, y2, x2]
  bboxes : [num_bboxes, 4]
  """
  return [bboxes[:,[1, 0, 3, 2]]]
  
def input_nodes(
  # An array of paths to tfrecords files
  tfrecords,
  
  # We'll pad all images to have this number of bounding boxes
  max_num_bboxes = 5, # This needs to be equal to the maximum number of bounding boxes for any image
  
  # Data augmentation depends on whether we are in train vs (test / eval) mode
  augment=True,

  # number of times to read the tfrecords
  num_epochs=None,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  shuffle_batch = True,
  capacity = 1000,
  min_after_dequeue = 96,

  # And tensorboard summaries of the images
  add_summaries=True,

  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs
    )

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64)
      }
    )

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32)
    image_id = features['image/id']
    
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    
    # combine the bounding boxes (the shape should be [bbox_coords, num_bboxes])
    bboxes = tf.concat(0, [xmin, ymin, xmax, ymax])
    # order the bboxes so that they have the shape: [num_bboxes, bbox_coords]
    bboxes = tf.transpose(bboxes, [1, 0])
    
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)
    
    
    if add_summaries:
      tf.image_summary('orig_image', tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.reshape(tf.py_func(reshape_bboxes, [bboxes], [tf.float32])[0], [1, -1, 4])))
    
    
    # This is where we will do some image augmentations
    # We need to do the same transformations to the bounding boxes.
    if augment:
      params = [image, bboxes, cfg.RANDOM_FLIP, cfg.RANDOM_BBOX_SHIFT, cfg.MAX_BBOX_COORD_SHIFT, cfg.RANDOM_CROP]
      output = tf.py_func(augment_image_and_bboxes, params, [tf.float32, tf.float32, tf.int32, tf.int32], name='augment_image')
      image = output[0]
      bboxes = output[1]
      
      # tf.resize_images requires the shapes to be defined.
      image = tf.reshape(image, tf.pack([output[2], output[3], 3]))
      
      if add_summaries:
        tf.image_summary('augmented_image', tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.reshape(tf.py_func(reshape_bboxes, [bboxes], [tf.float32])[0], [1, -1, 4])))
    
      
    # pad the number of boxes so that all images have `max_num_bboxes`
    # We could have had the generation code do this.
    num_rows_to_pad = tf.maximum(0, max_num_bboxes - num_bboxes)
    bboxes = tf.pad(bboxes, tf.pack([tf.pack([0, num_rows_to_pad]), [0, 0]]))
    
    if cfg.MAINTAIN_ASPECT_RATIO:
      # Resize the image up, then pad with 0s
      params = [image, tf.constant(cfg.INPUT_SIZE), tf.constant(cfg.INPUT_SIZE)]
      output = tf.py_func(resize_image_maintain_aspect_ratio, params, [tf.float32, tf.int32, tf.int32], name="resize_maintain_aspect_ratio")
      image = output[0]
      image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
      
      # GVH: We don't need to scale the bounding boxes when preserving the aspect ratio
      # but we do need to take into account the extra padding 
      
      new_height = output[1] 
      new_width = output[2]

      width_diff = tf.cast(cfg.INPUT_SIZE - new_width, tf.float32) / tf.cast(cfg.INPUT_SIZE, tf.float32)
      height_diff = tf.cast(cfg.INPUT_SIZE - new_height, tf.float32) / tf.cast(cfg.INPUT_SIZE, tf.float32)
      bboxes = bboxes * tf.cast(tf.pack([new_width, new_height, new_width, new_height]), tf.float32) - tf.cast(tf.pack([width_diff, height_diff, width_diff, height_diff]), tf.float32)
      bboxes = bboxes / tf.cast(tf.pack([cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE]), tf.float32)
    
    else:
      # Since the bboxes are already in normalized coordinates, we are good to go.
      # All we need to do is resize the image
      image = tf.image.resize_images(image, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
      image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    
    image -= cfg.IMAGE_MEAN
    image /= cfg.IMAGE_STD
    
    bboxes.set_shape([max_num_bboxes, 4])
    
    # Place the images on another queue that will be sampled by the model
    if shuffle_batch:
      images, batched_bboxes, batched_num_bboxes, image_ids = tf.train.shuffle_batch(
        [image, bboxes, num_bboxes, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
      )

    else:
      images, batched_bboxes, batched_num_bboxes, image_ids = tf.train.batch(
        [image, bboxes, num_bboxes, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=False
      )

  # return a batch of images and their labels
  return images, batched_bboxes, batched_num_bboxes, image_ids