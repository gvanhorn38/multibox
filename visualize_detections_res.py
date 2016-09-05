import argparse
import cPickle as pickle
import json
import logging
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from config import parse_config_file
from inputs import input_nodes
import model_res as model



def visualize(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    images, batched_bboxes, batched_num_bboxes, image_ids = input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes=cfg.MAX_NUM_BBOXES,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = False,
      augment=cfg.AUGMENT_IMAGE,
      shuffle_batch=False,
      capacity = cfg.QUEUE_CAPACITY,
      min_after_dequeue = cfg.QUEUE_MIN,
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
        num_bboxes_per_cell = 5,
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

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    sess = tf.Session(graph=graph, config=sess_config)
    
    with sess.as_default():

      fetches = [locations, confidences, images, batched_bboxes, batched_num_bboxes]
      
      coord = tf.train.Coordinator()
      
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      
      plt.ion()
      
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
        
        total_sample_count = 0
        step = 0
        done = False
        while not coord.should_stop() and not done:
          
          t = time.time()
          outputs = sess.run(fetches)
          dt = time.time()-t
          
          locs = outputs[0]
          confs = outputs[1]
          imgs = outputs[2]
          gt_bboxes = outputs[3]
          gt_num_bboxes = outputs[4]
          
          print locs.shape
          print confs.shape
          
          for b in range(cfg.BATCH_SIZE):
            
            # Show the image
            image = imgs[b]
            plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
            
            num_gt_bboxes_in_image = gt_num_bboxes[b]
            
            # Draw the GT Boxes in blue
            for i in range(num_gt_bboxes_in_image):
              gt_bbox = gt_bboxes[b][i]
              xmin, ymin, xmax, ymax = gt_bbox * cfg.INPUT_SIZE
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')

            # Draw the most confident boxes in red
            indices = np.argsort(confs[b].ravel())[::-1]
            for i, index in enumerate(indices[0:num_gt_bboxes_in_image]):
            
              loc = locs[b][index].ravel()
              conf = confs[b][index]
              prior = bbox_priors[index]
              
              print "Location: ", loc
              print "Prior: ", prior
              print "Index: ", index
              
              # Plot the predicted location in red
              xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
              
              # Plot the prior in yellow
              xmin, ymin, xmax, ymax = prior * cfg.INPUT_SIZE
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'g-')
              
              print "Pred Confidence for box %d: %f" % (i, conf)
              
            plt.show()
            
            
            t = raw_input("push button")
            if t != '':
              done = True
              break
            plt.clf()
            

      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)

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

  
  visualize(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    checkpoint_dir=args.checkpoint_dir,
    specific_model_path = args.specific_model,
    cfg=cfg
  ) 

if __name__ == '__main__':
  main()    
            