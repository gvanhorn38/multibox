import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import loss
import model_res as model

NUM_BBOX_LOCATIONS = 646

class InceptionResnetTest(tf.test.TestCase):

  def testBuildDetection(self):
    """Build the model.
    """
    
    graph = tf.get_default_graph()
    
    with graph.as_default(), self.test_session() as sess:

      # Just placeholders 
      images = tf.placeholder(tf.float32, [1, 299, 299, 3])
      batched_bboxes = tf.placeholder(tf.float32, [1, 5, 4])
      batched_num_bboxes = tf.placeholder(tf.int32, [1, 1])
      bbox_priors = tf.placeholder(tf.float32, [NUM_BBOX_LOCATIONS, 4])

      batch_norm_params = {
          'decay': 0.997,
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
          inputs = images,
          num_bboxes_per_cell = 5,
          reuse=False,
          scope=''
        )
        
        self.assertListEqual(locs.get_shape().as_list(),
                           [1, NUM_BBOX_LOCATIONS, 4])
        self.assertListEqual(confs.get_shape().as_list(),
                           [1, NUM_BBOX_LOCATIONS, 1])

  def testBuildLoss(self):
    """Build the model and add the loss.
    """
      
    graph = tf.get_default_graph()

    with graph.as_default(), self.test_session() as sess:

      # Just placeholders 
      images = tf.placeholder(tf.float32, [1, 299, 299, 3])
      batched_bboxes = tf.placeholder(tf.float32, [1, 5, 4])
      batched_num_bboxes = tf.placeholder(tf.int32, [1, 1])
      bbox_priors = tf.placeholder(tf.float32, [NUM_BBOX_LOCATIONS, 4])

      batch_norm_params = {
          'decay': 0.997,
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
          inputs = images,
          num_bboxes_per_cell = 5,
          reuse=False,
          scope=''
        )

      location_loss, confidence_loss = loss.add_loss(
        locations = locs, 
        confidences = confs, 
        batched_bboxes = batched_bboxes, 
        batched_num_bboxes = batched_num_bboxes, 
        bbox_priors = bbox_priors, 
        location_loss_alpha = 1.0
      )
      
      self.assertTrue(location_loss in graph.get_collection(tf.GraphKeys.LOSSES))
      self.assertTrue(confidence_loss in graph.get_collection(tf.GraphKeys.LOSSES))

  def testSingleBoundingBox(self):
    """Test an image with one bounding box.
    """
    
    graph = tf.get_default_graph()
    
    with graph.as_default(), self.test_session() as sess:

      # One ground truth bounding box
      images = tf.random_uniform([1, 299, 299, 3], minval=-1, maxval=1, dtype=tf.float32)
      batched_bboxes = np.zeros([1, 5, 4])
      batched_bboxes[0, 0] = np.array([0.1, 0.1, 0.9, 0.9])
      batched_num_bboxes = np.array([1])
      bbox_priors = tf.random_uniform([NUM_BBOX_LOCATIONS, 4], minval=0, maxval=1, dtype=tf.float32)

      batch_norm_params = {
          'decay': 0.997,
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
          inputs = images,
          num_bboxes_per_cell = 5,
          reuse=False,
          scope=''
        )

        location_loss, confidence_loss = loss.add_loss(
          locations = locs, 
          confidences = confs, 
          batched_bboxes = batched_bboxes, 
          batched_num_bboxes = batched_num_bboxes, 
          bbox_priors = bbox_priors, 
          location_loss_alpha = 1.0
        )

      total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
      
      sess.run(tf.initialize_all_variables())
      fetches = [location_loss, confidence_loss, total_loss]
      outputs = sess.run(fetches)
      
      self.assertTrue(outputs[0] > 0)
      self.assertTrue(outputs[1] > 0)
      self.assertTrue(outputs[0] + outputs[1] == outputs[2]) 

  def testNoGTBoundingBox(self):
    """A test where the image has no ground truth bounding boxes.
    """
    
    graph = tf.get_default_graph()

    with graph.as_default(), self.test_session() as sess:

      # No ground truth bounding boxes
      images = tf.random_uniform([1, 299, 299, 3], minval=-1, maxval=1, dtype=tf.float32)
      batched_bboxes = tf.zeros([1, 5, 4])
      batched_num_bboxes = np.array([0])
      bbox_priors = tf.random_uniform([NUM_BBOX_LOCATIONS, 4], minval=0, maxval=1, dtype=tf.float32)

      batch_norm_params = {
          'decay': 0.997,
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
          inputs = images,
          num_bboxes_per_cell = 5,
          reuse=False,
          scope=''
        )

        location_loss, confidence_loss = loss.add_loss(
          locations = locs, 
          confidences = confs, 
          batched_bboxes = batched_bboxes, 
          batched_num_bboxes = batched_num_bboxes, 
          bbox_priors = bbox_priors, 
          location_loss_alpha = 1.0
        )

      total_loss = slim.losses.get_total_loss(add_regularization_losses=False)

      sess.run(tf.initialize_all_variables())
      fetches = [location_loss, confidence_loss, total_loss]
      outputs = sess.run(fetches)
      
      self.assertTrue(outputs[0] == 0)
      self.assertTrue(outputs[1] > 0)
      self.assertTrue(outputs[0] + outputs[1] == outputs[2]) 

  def testBoxWithNoBox(self):
    """A test where one image has a bounding box, and another one does not.
    """
    
    graph = tf.get_default_graph()

    with graph.as_default(), self.test_session() as sess:

      # No ground truth bounding boxes
      images = tf.random_uniform([2, 299, 299, 3], minval=-1, maxval=1, dtype=tf.float32)
      batched_bboxes = np.zeros([2, 5, 4])
      batched_bboxes[0, 0] = np.array([0.1, 0.1, 0.9, 0.9])
      batched_num_bboxes = np.array([1, 0])
      bbox_priors = tf.random_uniform([NUM_BBOX_LOCATIONS, 4], minval=0, maxval=1, dtype=tf.float32)

      batch_norm_params = {
          'decay': 0.997,
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
          inputs = images,
          num_bboxes_per_cell = 5,
          reuse=False,
          scope=''
        )

        location_loss, confidence_loss = loss.add_loss(
          locations = locs, 
          confidences = confs, 
          batched_bboxes = batched_bboxes, 
          batched_num_bboxes = batched_num_bboxes, 
          bbox_priors = bbox_priors, 
          location_loss_alpha = 1.0
        )

      total_loss = slim.losses.get_total_loss(add_regularization_losses=False)

      sess.run(tf.initialize_all_variables())
      fetches = [location_loss, confidence_loss, total_loss]
      outputs = sess.run(fetches)
      
      self.assertTrue(outputs[0] > 0)
      self.assertTrue(outputs[1] > 0)
      self.assertTrue(outputs[0] + outputs[1] == outputs[2])
    
  
if __name__ == '__main__':
  tf.test.main()   

