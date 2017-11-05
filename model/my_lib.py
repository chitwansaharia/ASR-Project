import numpy as np
import sys
import tensorflow as tf

def get_scope_var(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope=scope_name)

def get_var_with_regex(scope_name, l_names):
    l_var = []
    all_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES , scope=scope_name)
    for i, v_name in enumerate([ v.name for v in all_vars]):
        for name in l_names:
            if name in v_name:
                l_var.append(all_vars[i])
                continue
    return l_var 

def data_type():
  return tf.float32

def human_readable(num, suffix=''):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1000.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_num_params(l_var):
    total_params = 0 
    for var in l_var:
        num_params = 1
        for dim in var.get_shape() : num_params *= dim.value
        print var.name, "->num_params: ", human_readable(num_params)
        total_params += num_params
    print "Total_params :" , human_readable(total_params)
    return total_params

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def print_same_line(*arg):
    string = "\r"
    for a in arg: string += " " + str(a)
    sys.stdout.write(string)
    sys.stdout.flush()

def is_checkpoint(i, total, step):
    batch = total // step
    l_ckp = [k * batch for k in range(step)] + [total] + [1]
    return (i in l_ckp)

def tf_batch_norm(x, n_out, phase_train, scope = None, \
                  reuse = None, axes = None, trainable = True):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor,
        axes :       List of axes to compute momemts,
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope, 'bn', reuse = reuse):
        beta = tf.get_variable('beta', [n_out], \
                        initializer = tf.constant_initializer(0.0),\
                        dtype = tf.float32, trainable = trainable)
        gamma = tf.get_variable('gamma', [n_out], \
                        initializer = tf.constant_initializer(1.0), \
                        dtype = tf.float32, trainable = trainable)
        if not axes:
            if len(x.get_shape()) == 4 : axes = [0, 1, 2]
            else : axes = [0]
        batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

