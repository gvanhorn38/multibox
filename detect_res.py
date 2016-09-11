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
        'image/encoded'  : tf.FixedLenFeature([], tf.string)
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
    
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [cfg.INPUT_SIZE, cfg.INPUT_SIZE],
                                      align_corners=False)
    image = tf.squeeze(image, [0])
    
    images, image_ids = tf.train.batch(
      [image, image_id],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity= capacity, 
      enqueue_many=False
    )

  # return a batch of images and their labels
  return images, image_ids

def detect(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, save_dir, max_detections, max_iterations, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    images, image_ids = input_nodes(
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
        inputs = images,
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

    fetches = [locations, confidences, image_ids]
    
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
          img_ids = outputs[2]
          
          for b in range(cfg.BATCH_SIZE):
            
            indices = np.argsort(confs[b].ravel())[::-1]
            img_id = img_ids[b]
            
            num_detections = 0

            for index in indices:
              loc = locs[b][index].ravel()
              conf = confs[b][index]          
              prior = bbox_priors[index]
              
              pred = np.clip(prior + loc, 0., 1.)
              #pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred
              
              # Not sure what we want to do here. Its interesting that we don't enforce this anywhere in the model
              # if pred_xmin > pred_xmax:
              #   t = pred_xmax
              #   pred_xmax = pred_xmin
              #   pred_xmin = t
              # if pred_ymin > pred_ymax:
              #   t = pred_ymax
              #   pred_ymax = pred_ymin
              #   pred_ymin = t
              # pred = np.array([pred_xmin, pred_ymin, pred_xmax, pred_xmin]).astype(np.float32)
              
              # Restrict the locations to be within the image 
              # This could cause some bounding boxes to have zero area
              # pred_xmin = float(max(0., pred_xmin))
              # pred_xmax = float(min(1., pred_xmax))
              # pred_ymin = float(max(0., pred_ymin))
              # pred_ymax = float(min(1., pred_ymax)) 
              
              # Ignore bounding boxes that have zero area 
              # if (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin) < 1e-8:
              #   continue

              detection_results.append({
                "image_id" : int(img_id), # converts  from np.array
                "bbox" : pred.tolist(), # [pred_xmin, pred_ymin, pred_xmax, pred_ymax],
                "score" : float(conf), # converts from np.array
              })

              # Stop early if we reach the max number of detections to save per image
              num_detections += 1
              if num_detections == max_detections:
                break

          step += 1
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)

          if max_iterations > 0 and step == max_iterations:
            break  

      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)
      
      # save the results
      save_path = os.path.join(save_dir, "results-%d.json" % global_step)
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
    