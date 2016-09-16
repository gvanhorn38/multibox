"""
For testing a detection system, we will take an approach similar to the COCO detection 
challenge. Detection results will be stored and 
"""
import argparse
import cPickle as pickle
import json
import logging
import numpy as np
import os
import pprint
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from config import parse_config_file
from inputs import resize_image_maintain_aspect_ratio
import model_res as model


from matplotlib import pyplot as plt
import numpy as np


def extract_patches(image, patch_dims, strides):
  """
  Args:
    image (np.array) : the image to extract the patches from
    patch_dims (tuple) : the (height, width) size of the patch to extract from the image (assumed to be square)
    strides (tuple) : the (y, x) stride of the patches (in height and width)
    
  Returns:
    list : the patches 
    list : offsets for each patch 
  """
  
  image_height, image_width = image.shape[:2]
  patch_height, patch_width = patch_dims
  h_stride, w_stride = strides
  
  patches = []
  patch_offsets = []
  
  for h in range(0,image_height-patch_height+1,h_stride):
    for w in range(0,image_width-patch_width+1,w_stride):
      
      p = image[h:h+patch_height, w:w+patch_width]
      patches.append(p)
      patch_offsets.append((h, w))
  
  patches = np.array(patches).astype(np.float32)
  patch_offsets = np.array(patch_offsets).astype(np.int32)

  # print image.shape
  # print patches.shape
  # print patch_offsets.shape
  # print 

  if patches.shape[0] == 0:
    patches = np.zeros([0, patch_height, patch_width, 3])
    patch_offsets = np.zeros([0, 2])

  return [patches, patch_offsets, np.int32(len(patches))]
      
def filter_proposals(bboxes, confidences):
  """We want to filter out proposals that are not completely contained in the square [.1, .1, .9, .9]
  
  Args: 
    bboxes np.array: proposed bboxes [x1, y1, x2, y2] in normalized coordinates
    confidences np.array: confidences for the proposed boxes
  
  Returns:
    np.array : the filtered bboxes
    np.array : the confidences for the bboxes
  """
  
  filtered_bboxes = []
  filtered_confidences = []
  
  for bbox, conf in zip(bboxes, confidences):
    if np.any(bbox < .1) or np.any(bbox > .9):
      continue
    filtered_bboxes.append(bbox)
    filtered_confidences.append(conf)
  
  return np.array(filtered_bboxes), np.array(filtered_confidences)

def convert_proposals(bboxes, offset, patch_dims, image_dims):
  """Convert the coordinates of the proposed bboxes to account for the offset of the patch
  
  Args:
    bboxes (np.array) : the proposed bboxes [x1, y1, x2, y2] in normalized coordinates
    offset (tuple) : the (y, x) offset of the patch in relation to the image
    patch_dims (tuple) : the (height, width) dimensions of the patch
    image_dims (tuple) : the (height, width) dimensions of the image
  Returns:
    np.array : the converted bounding boxes
  """
  
  x_scale = patch_dims[1] / float(image_dims[1])
  y_scale = patch_dims[0] / float(image_dims[0])
  
  x_offset = offset[1] / float(image_dims[1])
  y_offset = offset[0] / float(image_dims[0])
  
  converted_bboxes = bboxes * np.array([x_scale, y_scale, x_scale, y_scale]) + np.array([x_offset, y_offset, x_offset, y_offset])
  
  return converted_bboxes


def input_nodes(
  # An array of paths to tfrecords files
  tfrecords,

  # number of times to read the tfrecords
  num_epochs=1,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  capacity = 1000,
  
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
        'image/height' : tf.FixedLenFeature([], tf.int64),
        'image/width' : tf.FixedLenFeature([], tf.int64),
      }
    )
    
    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    
    # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_id = features['image/id']
    
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    
    image_height = features['image/height']
    image_width = features['image/width']

    # Extract the 299 x 299 patches
    params = [image, (299, 299), (149, 149)]
    output = tf.py_func(extract_patches, params, [tf.float32, tf.int32, tf.int32], name="extract_patches_299")
    patches_299 = output[0]
    num_patches_299 = output[2]
    patches_299.set_shape([None, 299, 299, 3])
    patches_299 = tf.cond(tf.greater(num_patches_299, 0), 
      lambda: tf.image.resize_images(patches_299, cfg.INPUT_SIZE, cfg.INPUT_SIZE, method=0, align_corners=False),
      lambda: tf.zeros([0, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    )
    patch_offsets_299 = output[1]
    patch_dims_299 = tf.tile([[299, 299]], [num_patches_299, 1])
    patch_restrict_299 = tf.tile([[1]], [num_patches_299, 1])
    patch_max_to_keep_299 = tf.tile([[5]], [num_patches_299, 1])

    # Extract the 185 x 185 patches
    params = [image, (185, 185), (91, 91)]
    output = tf.py_func(extract_patches, params, [tf.float32, tf.int32, tf.int32], name="extract_patches_185")
    patches_185 = output[0]
    num_patches_185 = output[2]
    patches_185.set_shape([None, 185, 185, 3])
    patches_185 = tf.cond(tf.greater(num_patches_299, 0), 
      lambda: tf.image.resize_images(patches_185, cfg.INPUT_SIZE, cfg.INPUT_SIZE, method=0, align_corners=False),
      lambda: tf.zeros([0, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    )
    patch_offsets_185 = output[1]
    patch_dims_185 = tf.tile([[185, 185]], [num_patches_185, 1])
    patch_restrict_185 = tf.tile([[1]], [num_patches_185, 1])
    patch_max_to_keep_185 = tf.tile([[2]], [num_patches_185, 1])

    # The actual image
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [cfg.INPUT_SIZE, cfg.INPUT_SIZE],
                                      align_corners=False)
    image_offsets = np.array([[0, 0]], dtype=np.int32)
    image_dims = tf.cast([[image_height, image_width]], dtype=tf.int32)
    image_restrict = np.array([[0]], dtype=np.int32)
    image_max_to_keep = np.array([[30]], dtype=np.int32)

    # We should probably do the flipped version too...

    # Combine the 299 and 185 patches. 
    total_patches = 1 + num_patches_299 + num_patches_185
    patches = tf.concat(0, [image, patches_299, patches_185])
    patch_offsets = tf.concat(0, [image_offsets, patch_offsets_299, patch_offsets_185])
    patch_dims = tf.concat(0, [image_dims, patch_dims_299, patch_dims_185])
    patch_restrict = tf.concat(0, [image_restrict, patch_restrict_299, patch_restrict_185])
    patch_max_to_keep = tf.concat(0, [image_max_to_keep, patch_max_to_keep_299, patch_max_to_keep_185])
    image_height_widths = tf.tile([[image_height, image_width]], [total_patches, 1])
    image_ids = tf.tile([[image_id]], [total_patches, 1])


    # Set the shape of everything for the queue
    patches.set_shape([None, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    patch_offsets.set_shape([None, 2])
    patch_dims.set_shape([None, 2])
    patch_restrict.set_shape([None, 1])
    patch_max_to_keep.set_shape([None, 1])
    image_height_widths.set_shape([None, 2])
    image_ids.set_shape([None, 1])

    batched_images, batched_offsets, batched_dims, batched_restrict, batched_max_to_keep, batched_heights_widths, batched_image_ids = tf.train.batch(
      [patches, patch_offsets, patch_dims, patch_restrict, patch_max_to_keep, image_height_widths, image_ids],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity= capacity, 
      enqueue_many=True
    )

  # return a batch of images and their labels
  return batched_images, batched_offsets, batched_dims, batched_restrict, batched_max_to_keep, batched_heights_widths, batched_image_ids

def detect(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, save_dir, max_detections, max_iterations, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    batched_images, batched_offsets, batched_dims, batched_restrict, batched_max_to_keep, batched_heights_widths, batched_image_ids = input_nodes(
      tfrecords=tfrecords,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      cfg=cfg
    )

    batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
      'is_training' : False
    }
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)):
      
      
      locations, confidences, inception_vars = model.build(
        inputs = batched_images,
        num_bboxes_per_cell = cfg.NUM_BBOXES_PER_CELL,
        reuse=False,
        scope=''
      )

    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
      for var in slim.get_model_variables()
    }
  
    
    # Restore the parameters
    saver = tf.train.Saver(shadow_vars, reshape=True)

    fetches = [locations, confidences, batched_offsets, batched_dims, batched_restrict, batched_max_to_keep, batched_heights_widths, batched_image_ids]
    
    coord = tf.train.Coordinator()

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    sess = tf.Session(graph=graph, config=sess_config)

    detection_results = []
    with sess.as_default():

      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      try:

        if specific_model_path == None:
          ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
          if ckpt and ckpt.model_checkpoint_path:
            specific_model_path = ckpt.model_checkpoint_path
          else:
            print('No checkpoint file found')
            return

        # Restores from checkpoint
        saver.restore(sess, specific_model_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(specific_model_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)
        
        print_str = ', '.join([
          'Step: %d',
          'Time/image (ms): %.1f'
        ])
        
        step = 0
        while not coord.should_stop():
          
          t = time.time()
          outputs = sess.run(fetches)
          dt = time.time()-t
          
          locs = outputs[0]
          confs = outputs[1]
          patch_offsets = outputs[2]
          patch_dims = outputs[3]
          patch_restrict = outputs[4]
          patch_max_to_keep = outputs[5]
          image_height_widths = outputs[6]
          image_ids = outputs[7]
          
          for b in range(cfg.BATCH_SIZE):
            
            # print "Patch Dims: ", patch_dims[b]
            # print "Patch Offset: ", patch_offsets[b]
            # print "Max to keep: ", patch_max_to_keep[b]
            # print "Filter preds: ", patch_restrict[b]
            # print "Image HxW: ", image_height_widths[b]
            # print

            img_id = int(np.asscalar(image_ids[b]))
            
            predicted_bboxes = locs[b] + bbox_priors
            predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)
            predicted_confs = confs[b]
            
            # Keep only the predictions that are completely contained in the [0.1, 0.1, 0.9, 0.9] square
            # for this patch
            if patch_restrict[b]:
              filtered_bboxes, filtered_confs = filter_proposals(predicted_bboxes, predicted_confs) 
            else:
              filtered_bboxes = predicted_bboxes
              filtered_confs = predicted_confs

            # No valid predictions? 
            if filtered_bboxes.shape[0] == 0:
              continue
            
            # Lets get rid of some of the predictions
            num_preds_to_keep = patch_max_to_keep[b]
            sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
            sorted_idxs = sorted_idxs[:num_preds_to_keep]
            filtered_bboxes = filtered_bboxes[sorted_idxs]
            filtered_confs = filtered_confs[sorted_idxs]
            
            # Convert the bounding boxes to the original image dimensions
            converted_bboxes = convert_proposals(
              bboxes = filtered_bboxes, 
              offset = patch_offsets[b], 
              patch_dims = patch_dims[b], 
              image_dims = image_height_widths[b]
            )
            
            for k in range(converted_bboxes.shape[0]):  
              detection_results.append({
                "image_id" : img_id, # converts  from np.array
                "bbox" : converted_bboxes[k].tolist(), # [pred_xmin, pred_ymin, pred_xmax, pred_ymax],
                "score" : float(np.asscalar(filtered_confs[k])), # converts from np.array
              })

          step += 1
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)

          if max_iterations > 0 and step == max_iterations:
            break  

      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)
      
      # image_id_count = {}
      # for r in detection_results:
      #   image_id_count.setdefault(r['image_id'], []).append(r['image_id'])
      
      # for image_id, c in image_id_count.items():
      #   print "%d : %d" % (image_id, len(c))

      # save the results
      save_path = os.path.join(save_dir, "results-dense-%d.json" % global_step)
      with open(save_path, 'w') as f: 
        json.dump(detection_results, f)

def parse_args():

    parser = argparse.ArgumentParser(description='Detect objects using a pretrained Multibox model')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
    
    parser.add_argument('--model', dest='specific_model',
                          help='path to a specific model to test against. This has precedence over the checkpoint_dir argument.', type=str,
                          required=False, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)
    
    parser.add_argument('--max_detections', dest='max_detections',
                        help='Maximum number of detection to store per image',
                        required=False, type=int, default=100) 
    
    parser.add_argument('--save_dir', dest='save_dir',
                        help='Directory to save the json result file.',
                        required=True, type=str)
    
    # parser.add_argument('--dense', dest='dense',
    #                     help='For each image, extract and process crops from the image.',
    #                     action='store_true', default=False)                

    args = parser.parse_args()
    
    if args.checkpoint_dir == None and args.specific_model == None:
      print "Either a checkpoint directory or a specific model needs to be specified."
      parser.print_help()
      sys.exit(1)
    
    return args

def main():
  args = parse_args()
  print "Command line arguments:"
  pprint.pprint(vars(args))
  print

  cfg = parse_config_file(args.config_file)
  print "Configurations:"
  pprint.pprint(cfg)
  print 
    
  with open(args.priors) as f:
    bbox_priors = pickle.load(f)
  bbox_priors = np.array(bbox_priors).astype(np.float32)

  # test_func = test.test
  # if args.dense:
  #   test_func = dense_test.test
  
  detect(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    checkpoint_dir=args.checkpoint_dir,
    specific_model_path = args.specific_model,
    save_dir = args.save_dir,
    max_detections = args.max_detections,
    max_iterations = args.max_iterations,
    cfg=cfg,
    
  ) 

if __name__ == '__main__':
  main()
    