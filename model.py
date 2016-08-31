import tensorflow as tf
import tensorflow.contrib.slim as slim

def build_base(inputs,
          is_training=True,
          scope=''):
  
  end_points = {}
  
  with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
    # 299 x 299 x 3
    end_points['conv0'] = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                      scope='conv0')
    # 149 x 149 x 32
    end_points['conv1'] = slim.conv2d(end_points['conv0'], 32, [3, 3],
                                      scope='conv1')
    # 147 x 147 x 32
    end_points['conv2'] = slim.conv2d(end_points['conv1'], 64, [3, 3],
                                      padding='SAME', scope='conv2')
    # 147 x 147 x 64
    end_points['pool1'] = slim.max_pool2d(end_points['conv2'], [3, 3],
                                        stride=2, scope='pool1')
    # 73 x 73 x 64
    end_points['conv3'] = slim.conv2d(end_points['pool1'], 80, [1, 1],
                                      scope='conv3')
    # 73 x 73 x 80.
    end_points['conv4'] = slim.conv2d(end_points['conv3'], 192, [3, 3],
                                      scope='conv4')
    # 71 x 71 x 192.
    end_points['pool2'] = slim.max_pool2d(end_points['conv4'], [3, 3],
                                        stride=2, scope='pool2')
    # 35 x 35 x 192.
    net = end_points['pool2']
  
  # Inception blocks
  with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
    # mixed: 35 x 35 x 256.
    with tf.variable_scope('mixed_35x35x256a'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 64, [1, 1])
      with tf.variable_scope('branch5x5'):
        branch5x5 = slim.conv2d(net, 48, [1, 1])
        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
      net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
      end_points['mixed_35x35x256a'] = net
    # mixed_1: 35 x 35 x 288.
    with tf.variable_scope('mixed_35x35x288a'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 64, [1, 1])
      with tf.variable_scope('branch5x5'):
        branch5x5 = slim.conv2d(net, 48, [1, 1])
        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
      net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
      end_points['mixed_35x35x288a'] = net
    # mixed_2: 35 x 35 x 288.
    with tf.variable_scope('mixed_35x35x288b'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 64, [1, 1])
      with tf.variable_scope('branch5x5'):
        branch5x5 = slim.conv2d(net, 48, [1, 1])
        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
      net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
      end_points['mixed_35x35x288b'] = net
    # mixed_3: 17 x 17 x 768.
    with tf.variable_scope('mixed_17x17x768a'):
      with tf.variable_scope('branch3x3'):
        branch3x3 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                  stride=2, padding='VALID')
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
      net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
      end_points['mixed_17x17x768a'] = net
    # mixed4: 17 x 17 x 768.
    with tf.variable_scope('mixed_17x17x768b'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 192, [1, 1])
      with tf.variable_scope('branch7x7'):
        branch7x7 = slim.conv2d(net, 128, [1, 1])
        branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
      with tf.variable_scope('branch7x7dbl'):
        branch7x7dbl = slim.conv2d(net, 128, [1, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
      end_points['mixed_17x17x768b'] = net
    # mixed_5: 17 x 17 x 768.
    with tf.variable_scope('mixed_17x17x768c'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 192, [1, 1])
      with tf.variable_scope('branch7x7'):
        branch7x7 = slim.conv2d(net, 160, [1, 1])
        branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
      with tf.variable_scope('branch7x7dbl'):
        branch7x7dbl = slim.conv2d(net, 160, [1, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
      end_points['mixed_17x17x768c'] = net
    # mixed_6: 17 x 17 x 768.
    with tf.variable_scope('mixed_17x17x768d'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 192, [1, 1])
      with tf.variable_scope('branch7x7'):
        branch7x7 = slim.conv2d(net, 160, [1, 1])
        branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
      with tf.variable_scope('branch7x7dbl'):
        branch7x7dbl = slim.conv2d(net, 160, [1, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
      end_points['mixed_17x17x768d'] = net
    # mixed_7: 17 x 17 x 768.
    with tf.variable_scope('mixed_17x17x768e'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 192, [1, 1])
      with tf.variable_scope('branch7x7'):
        branch7x7 = slim.conv2d(net, 192, [1, 1])
        branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
      with tf.variable_scope('branch7x7dbl'):
        branch7x7dbl = slim.conv2d(net, 192, [1, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
      end_points['mixed_17x17x768e'] = net
  
    # mixed_8: 8 x 8 x 1280.
    # Note that the scope below is not changed to not void previous
    # checkpoints.
    # (TODO) Fix the scope when appropriate.
    with tf.variable_scope('mixed_17x17x1280a'):
      with tf.variable_scope('branch3x3'):
        branch3x3 = slim.conv2d(net, 192, [1, 1])
        branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2,
                                padding='VALID')
      with tf.variable_scope('branch7x7x3'):
        branch7x7x3 = slim.conv2d(net, 192, [1, 1])
        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3],
                                  stride=2, padding='VALID')
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
      net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
      end_points['mixed_17x17x1280a'] = net
    # mixed_9: 8 x 8 x 2048.
    with tf.variable_scope('mixed_8x8x2048a'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 320, [1, 1])
      with tf.variable_scope('branch3x3'):
        branch3x3 = slim.conv2d(net, 384, [1, 1])
        branch3x3 = tf.concat(3, [slim.conv2d(branch3x3, 384, [1, 3]),
                                  slim.conv2d(branch3x3, 384, [3, 1])])
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 448, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
        branch3x3dbl = tf.concat(3, [slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                      slim.conv2d(branch3x3dbl, 384, [3, 1])])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
      end_points['mixed_8x8x2048a'] = net
    # mixed_10: 8 x 8 x 2048.
    with tf.variable_scope('mixed_8x8x2048b'):
      with tf.variable_scope('branch1x1'):
        branch1x1 = slim.conv2d(net, 320, [1, 1])
      with tf.variable_scope('branch3x3'):
        branch3x3 = slim.conv2d(net, 384, [1, 1])
        branch3x3 = tf.concat(3, [slim.conv2d(branch3x3, 384, [1, 3]),
                                  slim.conv2d(branch3x3, 384, [3, 1])])
      with tf.variable_scope('branch3x3dbl'):
        branch3x3dbl = slim.conv2d(net, 448, [1, 1])
        branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
        branch3x3dbl = tf.concat(3, [slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                      slim.conv2d(branch3x3dbl, 384, [3, 1])])
      with tf.variable_scope('branch_pool'):
        branch_pool = slim.avg_pool2d(net, [3, 3])
        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
      net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
      end_points['mixed_8x8x2048b'] = net
      
      return net, end_points

def build_detection_heads(inputs, num_bboxes_per_cell, is_training=True, scope=''):
  
  endpoints = {}
  
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d], stride=1, padding='SAME'):

    # 8 x 8 grid cells
    with tf.variable_scope("8x8"):
      # 8 x 8 x 2048 
      branch8x8 = slim.conv2d(inputs, 96, [1, 1])
      # 8 x 8 x 96
      branch8x8 = slim.conv2d(branch8x8, 96, [3, 3])
      # 8 x 8 x 96
      endpoints['8x8_locations'] = slim.conv2d(branch8x8, num_bboxes_per_cell * 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 8 x 8 x 96
      endpoints['8x8_confidences'] = slim.conv2d(branch8x8, num_bboxes_per_cell, [1, 1],
        activation_fn=None, normalizer_fn=None
      )

    # 6 x 6 grid cells
    with tf.variable_scope("6x6"):
      # 8 x 8 x 2048 
      branch6x6 = slim.conv2d(inputs, 96, [3, 3])
      # 8 x 8 x 96
      branch6x6 = slim.conv2d(branch6x6, 96, [3, 3], padding = "VALID")
      # 6 x 6 x 96
      endpoints['6x6_locations'] = slim.conv2d(branch6x6, num_bboxes_per_cell * 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 6 x 6 x 96
      endpoints['6x6_confidences'] = slim.conv2d(branch6x6, num_bboxes_per_cell, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
    
    # 8 x 8 x 2048
    net = slim.conv2d(inputs, 256, [3, 3], stride=2)

    # 4 x 4 grid cells
    with tf.variable_scope("4x4"):
      # 4 x 4 x 256
      branch4x4 = slim.conv2d(net, 128, [3, 3])
      # 4 x 4 x 128
      endpoints['4x4_locations'] = slim.conv2d(branch4x4, num_bboxes_per_cell * 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 4 x 4 x 128
      endpoints['4x4_confidences'] = slim.conv2d(branch4x4, num_bboxes_per_cell, [1, 1],
        activation_fn=None, normalizer_fn=None
      )

    # 3 x 3 grid cells
    with tf.variable_scope("3x3"):
      # 4 x 4 x 256
      branch3x3 = slim.conv2d(net, 128, [1, 1])
      # 4 x 4 x 128
      branch3x3 = slim.conv2d(branch3x3, 96, [2, 2], padding="VALID")
      # 3 x 3 x 96
      endpoints['3x3_locations'] = slim.conv2d(branch3x3, num_bboxes_per_cell * 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 3 x 3 x 96
      endpoints['3x3_confidences'] = slim.conv2d(branch3x3, num_bboxes_per_cell, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      
    # 2 x 2 grid cells
    with tf.variable_scope("2x2"):
      # 4 x 4 x 256
      branch2x2 = slim.conv2d(net, 128, [1, 1])
      # 4 x 4 x 128
      branch2x2 = slim.conv2d(branch2x2, 96, [3, 3], padding = "VALID")
      # 2 x 2 x 96
      endpoints['2x2_locations'] = slim.conv2d(branch2x2, num_bboxes_per_cell * 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 2 x 2 x 96
      endpoints['2x2_confidences'] = slim.conv2d(branch2x2, num_bboxes_per_cell, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      
    # 1 x 1 grid cell
    with tf.variable_scope("1x1"):
      # 8 x 8 x 2048
      branch1x1 = slim.avg_pool2d(inputs, [8, 8], padding="VALID")
      # 1 x 1 x 2048
      endpoints['1x1_locations'] = slim.conv2d(branch1x1, 4, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
      # 1 x 1 x 2048
      endpoints['1x1_confidences'] = slim.conv2d(branch1x1, 1, [1, 1],
        activation_fn=None, normalizer_fn=None
      )
    
    batch_size = inputs.get_shape().as_list()[0]

    # reshape the locations and confidences for easy concatenation
    detect_8_locations = tf.reshape(endpoints['8x8_locations'], [batch_size, -1])
    detect_8_confidences = tf.reshape(endpoints['8x8_confidences'], [batch_size, -1])

    detect_6_locations = tf.reshape(endpoints['6x6_locations'], [batch_size, -1])
    detect_6_confidences = tf.reshape(endpoints['6x6_confidences'], [batch_size, -1])

    detect_4_locations = tf.reshape(endpoints['4x4_locations'], [batch_size, -1])
    detect_4_confidences = tf.reshape(endpoints['4x4_confidences'], [batch_size, -1])

    detect_3_locations = tf.reshape(endpoints['3x3_locations'], [batch_size, -1])
    detect_3_confidences = tf.reshape(endpoints['3x3_confidences'], [batch_size, -1])

    detect_2_locations = tf.reshape(endpoints['2x2_locations'], [batch_size, -1])
    detect_2_confidences = tf.reshape(endpoints['2x2_confidences'], [batch_size, -1])

    detect_1_locations = tf.reshape(endpoints['1x1_locations'], [batch_size, -1])
    detect_1_confidences = tf.reshape(endpoints['1x1_confidences'], [batch_size, -1])    
          
    # Collect all of the locations and confidences 
    locations = tf.concat(1, [detect_8_locations, detect_6_locations, detect_4_locations, detect_3_locations, detect_2_locations, detect_1_locations])
    locations = tf.reshape(locations, [batch_size, -1, 4])
    
    confidences = tf.concat(1, [detect_8_confidences, detect_6_confidences, detect_4_confidences, detect_3_confidences, detect_2_confidences, detect_1_confidences])
    confidences = tf.reshape(confidences, [batch_size, -1, 1])
    confidences = tf.sigmoid(confidences)
    
  return locations, confidences, endpoints

def _name_in_checkpoint(var):
  if "fully_connected" in var.op.name:
    return var.op.name.replace("fully_connected", "FC")
  return var.op.name

def build(inputs, num_bboxes_per_cell, scope=''):

  with tf.op_scope([inputs], scope, 'multibox'):
    
    # Build the Inception-v3 model
    features, _ = build_base(inputs)
    
    # Save off the original variables (for ease of restoring)
    model_variables = slim.get_model_variables()
    original_inception_vars = {_name_in_checkpoint(var):var for var in model_variables}

    # Add on the detection heads
    locs, confs, _ = build_detection_heads(features, 5)
  
  return locs, confs, original_inception_vars
