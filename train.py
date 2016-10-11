import argparse
import copy
import cPickle as pickle
import logging
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import parse_config_file
import inputs
import loss
import model

def build_fully_trainable_model(inputs, cfg):

  batch_norm_params = {
    'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
    'epsilon': 0.001,
    'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
    'is_training' : True
  }
  # Set activation_fn and parameters for batch_norm.
  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.00004),
                      biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
    
    locs, confs, inception_vars = model.build(
      inputs = inputs,
      num_bboxes_per_cell = cfg.NUM_BBOXES_PER_CELL,
      reuse=False,
      scope=''
    )

  return locs, confs, inception_vars

def build_finetunable_model(inputs, cfg):

  with slim.arg_scope([slim.conv2d], 
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_regularizer=slim.l2_regularizer(0.00004),
                      biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
      
      batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [],
        'is_training' : False
      }
      with slim.arg_scope([slim.conv2d], normalizer_params=batch_norm_params):
        features, _ = model.inception_resnet_v2(inputs, reuse=False, scope='InceptionResnetV2')
        
      # Save off the original variables (for ease of restoring)
      model_variables = slim.get_model_variables()
      inception_vars = {var.op.name:var for var in model_variables}

      batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : True
      }
      with slim.arg_scope([slim.conv2d], normalizer_params=batch_norm_params):

        # Add on the detection heads
        locs, confs, _ = model.build_detection_heads(features, cfg.NUM_BBOXES_PER_CELL)
        model_variables = slim.get_model_variables()
        detection_vars = {var.op.name:var for var in model_variables if var.op.name not in inception_vars}
  
  return locs, confs, inception_vars, detection_vars

def train(tfrecords, bbox_priors, logdir, cfg, pretrained_model_path=None, fine_tune=False):
  """
  Args:
    tfrecords (list)
    bbox_priors (np.array)
    logdir (str)
    cfg (EasyDict)
    pretrained_model_path (str) : path to a pretrained Inception Network
  """
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():
    
    # Create a variable to count the number of train() calls. 
    global_step = slim.get_or_create_global_step()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cfg.NUM_TRAIN_EXAMPLES /
                             cfg.BATCH_SIZE)
    decay_steps = int(num_batches_per_epoch * cfg.NUM_EPOCHS_PER_DELAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
      learning_rate=cfg.INITIAL_LEARNING_RATE,
      global_step=global_step,
      decay_steps=decay_steps,
      decay_rate=cfg.LEARNING_RATE_DECAY_FACTOR,
      staircase=cfg.LEARNING_RATE_STAIRCASE
    )

    # Create an optimizer that performs gradient descent.
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=lr,
      decay=cfg.RMSPROP_DECAY,
      momentum=cfg.RMSPROP_MOMENTUM,
      epsilon=cfg.RMSPROP_EPSILON
    )

    batched_images, batched_bboxes, batched_num_bboxes, image_ids = inputs.input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes = cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity=cfg.QUEUE_CAPACITY,
      min_after_dequeue=cfg.QUEUE_MIN,
      add_summaries = True,
      shuffle_batch=True,
      cfg=cfg
    )
    
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
    if fine_tune: 
      locs, confs, inception_vars, detection_vars = build_finetunable_model(batched_images, cfg)
      all_trainable_var_names = [v.op.name for v in tf.trainable_variables()]
      trainable_vars = [v for v_name, v in detection_vars.items() if v_name in all_trainable_var_names]
    else:
      locs, confs, inception_vars = build_fully_trainable_model(batched_images, cfg)
      trainable_vars = tf.trainable_variables()

    location_loss, confidence_loss = loss.add_loss(
      locations = locs, 
      confidences = confs, 
      batched_bboxes = batched_bboxes, 
      batched_num_bboxes = batched_num_bboxes, 
      bbox_priors = bbox_priors, 
      location_loss_alpha = cfg.LOCATION_LOSS_ALPHA
    )
    
    total_loss = slim.losses.get_total_loss()

    # Track the moving averages of all trainable variables.
    # At test time we'll restore all variables with the average value
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY,
      num_updates=global_step
    )
    variables_to_average = (slim.get_model_variables()) # Makes it easier to restore for eval and detect purposes (whether you use the fine_tune flag or not)
    maintain_averages_op = ema.apply(variables_to_average)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_averages_op)

    train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=trainable_vars)

    # Summary operations
    summary_op = tf.merge_summary([
      tf.scalar_summary('total_loss', total_loss),
      tf.scalar_summary('location_loss', location_loss),
      tf.scalar_summary('confidence_loss', confidence_loss),
      tf.scalar_summary('learning_rate', lr)
    ] + input_summaries)

    if pretrained_model_path != None:
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(pretrained_model_path, inception_vars)
    else:
      init_assign_op = tf.no_op()
      init_feed_dict = {}

    # Create an initial assignment function.
    def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )

    saver = tf.train.Saver(
      # Save all variables
      max_to_keep = cfg.MAX_TO_KEEP,
      keep_checkpoint_every_n_hours = cfg.KEEP_CHECKPOINT_EVERY_N_HOURS
    )

    # Run training.
    slim.learning.train(train_op, logdir, 
      init_fn=InitAssignFn,
      number_of_steps=cfg.NUM_TRAIN_ITERATIONS,
      save_summaries_secs=cfg.SAVE_SUMMARY_SECS,
      save_interval_secs=cfg.SAVE_INTERVAL_SECS,
      saver=saver,
      session_config=sess_config,
      summary_op = summary_op,
      log_every_n_steps = cfg.LOG_EVERY_N_STEPS
    )

def parse_args():

    parser = argparse.ArgumentParser(description='Train the multibox detection system')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files that contain the training data', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--logdir', dest='logdir',
                          help='path to directory to store summary files and checkpoint files', type=str,
                          required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        help='Is this the first iteration? If so pass a full path to a pretrained Inception-v3 model.',
                        required=False, type=str, default=None)
    
    parser.add_argument('--fine_tune', dest='fine_tune',
                        help='If True, then only the variables in the detection heads will be trained, as opposed to the whole network.',
                        action='store_true', default=False)

    args = parser.parse_args()
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

  train(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    logdir=args.logdir,
    cfg=cfg,
    pretrained_model_path=args.pretrained_model,
    fine_tune = args.fine_tune
  )

if __name__ == '__main__':
  main()