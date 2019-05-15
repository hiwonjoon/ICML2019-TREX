import tensorflow as tf

class Conv2d(object) :
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW',padding='SAME') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
            self.padding = padding
    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( self.data_format =='NCHW' ) :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,
                                    use_cudnn_on_gpu=True,data_format='NCHW',
                                    strides=self.strides, padding=self.padding),
                        b,data_format='NCHW',name=name)
        else :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,data_format='NHWC',
                                    strides=self.strides, padding=self.padding),
                        b,data_format='NHWC',name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class WeightNormConv2d(object):
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NHWC',padding='SAME',epsilon=1e-9) :
        with tf.variable_scope(name) :
            assert data_format == 'NHWC'
            self.v = tf.get_variable('v', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.g = tf.get_variable('g',[output_dim],
                                     initializer=tf.constant_initializer(float('nan')))
            self.b = tf.get_variable('b',[output_dim],
                                     initializer=tf.constant_initializer(float('nan')))

            self.strides = [1, d_h, d_w, 1]
            self.padding = padding

            self.epsilon = epsilon

    def __call__(self,input_var,name=None,**kwargs) :
        def _init():
            v_norm = tf.nn.l2_normalize(self.v,axis=[0,1,2])
            t = tf.nn.conv2d(input_var,v_norm,self.strides,self.padding,data_format='NHWC')
            mu,var = tf.nn.moments(t,axes=[0,1,2])
            std = tf.sqrt(var+self.epsilon)
            return [tf.assign(self.g,1/std),tf.assign(self.b,-1.*mu/std)]

        require_init = tf.reduce_any(tf.is_nan(self.g))
        init_ops = tf.cond(require_init,_init,lambda : [self.g,self.b])

        with tf.control_dependencies(init_ops):
            w = tf.reshape(self.g,[1,1,1,tf.shape(self.v)[-1]]) * tf.nn.l2_normalize(self.v,axis=[0,1,2])
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,data_format='NHWC',
                                    strides=self.strides, padding=self.padding),
                        self.b,data_format='NHWC',name=name)

    def get_variables(self):
        #TODO: self.v should be l2-normalized or not? / currently not.
        return {'v':self.v,'b':self.b,'g':self.g}

class DepthConv2d(object) :
    def __init__(self,name,input_dim,channel_multiplier,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW', padding='SAME') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, channel_multiplier],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[input_dim*channel_multiplier], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
            self.padding = padding
    def __call__(self,input_var,name=None,**xargs) :
        return tf.nn.bias_add(
                    tf.nn.depthwise_conv2d(input_var, self.w,
                                data_format=self.data_format,
                                strides=self.strides, padding=self.padding),
                    self.b,data_format=self.data_format,name=name)

class Conv3d(object) :
    def __init__(self,name,input_dim,output_dim,k_t=2,k_h=4,k_w=4,d_t=1,d_h=1,d_w=1,
                 stddev=0.02, data_format='NDHWC') :
        with tf.variable_scope(name) :
            assert(data_format == 'NDHWC')
            self.w = tf.get_variable('w', [k_t, k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            self.strides = [d_t,d_h,d_w]
    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b
        #k_t,k_h,k_w,_,_ = self.w.get_shape().as_list()
        #_t = tf.pad(input_var, [[0,0],[0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0]], "SYMMETRIC")
        return tf.nn.bias_add(
                    tf.nn.convolution(input_var, w,
                                      strides=self.strides,
                                      data_format='NDHWC',
                                      padding='SAME'),
                    b,name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class DilatedConv3D(object) :
    def __init__(self,name,input_dim,output_dim,k_t=2,k_h=3,k_w=3,d_t=2,d_h=1,d_w=1,
                 stddev=0.02, data_format='NDHWC') :
        with tf.variable_scope(name) :
            assert(data_format == 'NDHWC')
            self.w = tf.get_variable('w', [k_t, k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            self.strides = [1,1,1]
            self.dilates = [d_t, d_h, d_w]
    def __call__(self,input_var,name=None) :
        k_t,k_h,k_w,_,_ = self.w.get_shape().as_list()
        _t = tf.pad(input_var, [[0,0],[0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0]], "SYMMETRIC")
        return tf.nn.bias_add(
                    tf.nn.convolution(_t, self.w,
                                      strides=self.strides, dilation_rate=self.dilates,
                                      padding='VALID'),
                    self.b,name=name)

class Linear(object) :
    def __init__(self,name,input_dim,output_dim,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w',[input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim],
                                initializer=tf.constant_initializer(0.0))

    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( input_var.shape.ndims > 2 ) :
            dims = tf.reduce_prod(tf.shape(input_var)[1:])
            return tf.matmul(tf.reshape(input_var,[-1,dims]),w) + b
        else :
            return tf.matmul(input_var,w)+b
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class WeightNormLinear(object):
    def __init__(self,name,input_dim,output_dim,stddev=0.02,epsilon=1e-10) :
        with tf.variable_scope(name) :
            self.v = tf.get_variable('v',[input_dim, output_dim],
                                     initializer=tf.random_normal_initializer(stddev=stddev))
            self.g = tf.get_variable('g',[output_dim],
                                     initializer=tf.constant_initializer(float('nan')))
            self.b = tf.get_variable('b',[output_dim],
                                     initializer=tf.constant_initializer(float('nan')))
            self.epsilon = epsilon

    def __call__(self,input_var,name=None,**kwargs) :
        if( input_var.shape.ndims > 2 ) :
            dims = tf.reduce_prod(tf.shape(input_var)[1:])
            input_var = tf.reshape(input_var,[-1,dims])

        def _init():
            v_norm = tf.nn.l2_normalize(self.v,axis=0)
            t = tf.matmul(input_var,v_norm)
            mu,var = tf.nn.moments(t,axes=[0])
            std = tf.sqrt(var+self.epsilon)
            return [tf.assign(self.g,1/std),tf.assign(self.b,-1.*mu/std)]

        require_init = tf.reduce_any(tf.is_nan(self.g))
        init_ops = tf.cond(require_init,_init,lambda : [self.g,self.b])

        with tf.control_dependencies(init_ops):
            w = tf.expand_dims(self.g,axis=0) * tf.nn.l2_normalize(self.v,axis=0)
            return tf.matmul(input_var,w)+self.b

    def get_variables(self):
        #TODO: self.v should be l2-normalized or not? / currently not.
        return {'v':self.v,'b':self.b,'g':self.g}

class SymPadConv2d(object): #Resize and Convolution(upsacle by 2)
    def __init__(self,name,input_dim,output_dim,
                 k_h=3,k_w=3,stddev=0.02) :
        assert k_h%2==1 and k_w%2==1, 'kernel size should be odd numbers to ensure exact size'
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))

        self.padding = [ [0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0] ]

    def __call__(self,input_var,name=None,**kwargs):
        _,h,w,c = input_var.shape.as_list()
        _t = tf.image.resize_nearest_neighbor(input_var, [h*2, w*2])
        _t = tf.pad(_t,self.padding, mode='SYMMETRIC')
        return tf.nn.bias_add(
                    tf.nn.conv2d(_t, self.w,
                                 data_format='NHWC', #we can't use cudnn due to resize method...
                                 strides=[1,1,1,1], padding="VALID"),
                    self.b,data_format='NHWC',name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class WeightNormSymPadConv2d(object): #Resize and Convolution(upsacle by 2)
    def __init__(self,name,input_dim,output_dim,
                 k_h=3,k_w=3,stddev=0.02) :
        assert k_h%2==1 and k_w%2==1, 'kernel size should be odd numbers to ensure exact size'
        with tf.variable_scope(name) :
            self.conv2d = WeightNormConv2d('conv',input_dim,output_dim,k_h,k_w,1,1,data_format='NHWC',padding='VALID')
        self.padding = [ [0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0] ]

    def __call__(self,input_var,name=None,**kwargs):
        _,h,w,c = input_var.shape.as_list()
        _t = tf.image.resize_nearest_neighbor(input_var, [h*2, w*2])
        _t = tf.pad(_t,self.padding, mode='SYMMETRIC')
        return self.conv2d(_t)

    def get_variables(self):
        return self.conv2d.get_variables()

class TransposedConv2d(object):
    def __init__(self,name,input_dim,out_dim,
                 k_h=4,k_w=4,d_h=2,d_w=2,stddev=0.02,data_format='NCHW') :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, out_dim, input_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[out_dim], initializer=tf.constant_initializer(0.0))

        self.data_format = data_format
        if( data_format =='NCHW' ):
            self.strides = [1, 1, d_h, d_w]
        else:
            self.strides = [1, d_h, d_w, 1]

    def __call__(self,input_var,name=None,**xargs):
        shapes = tf.shape(input_var)
        if( self.data_format == 'NCHW' ):
            shapes = tf.stack([shapes[0],tf.shape(self.b)[0],shapes[2]*self.strides[2],shapes[3]*self.strides[3]])
        else:
            shapes = tf.stack([shapes[0],shapes[1]*self.strides[1],shapes[2]*self.strides[2],tf.shape(self.b)[0]])

        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_var,self.w,output_shape=shapes,
                                data_format=self.data_format,
                                strides=self.strides,padding='SAME'),
            self.b,data_format=self.data_format,name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class WeightNormTransposedConv2d(object):
    def __init__(self,name,input_dim,out_dim,
                 k_h=4,k_w=4,d_h=2,d_w=2,stddev=0.02,data_format='NHWC',epsilon=1e-9) :
        with tf.variable_scope(name) :
            assert data_format == 'NHWC'
            self.v = tf.get_variable('v', [k_h, k_w, out_dim, input_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.g = tf.get_variable('g',[out_dim],
                                     initializer=tf.constant_initializer(float('nan')))
            self.b = tf.get_variable('b',[out_dim],
                                     initializer=tf.constant_initializer(float('nan')))

            self.strides = [1, d_h, d_w, 1]

            self.epsilon = epsilon

    def __call__(self,input_var,name=None,**kwargs) :
        shapes = tf.shape(input_var)
        shapes = tf.stack([shapes[0],shapes[1]*self.strides[1],shapes[2]*self.strides[2],tf.shape(self.b)[0]])

        def _init():
            v_norm = tf.nn.l2_normalize(self.v,axis=[0,1,3])
            t = tf.nn.conv2d_transpose(input_var,v_norm,
                                       output_shape=shapes,
                                       strides=self.strides,
                                       padding='SAME',
                                       data_format='NHWC')
            mu,var = tf.nn.moments(t,axes=[0,1,2])
            std = tf.sqrt(var+self.epsilon)
            return [tf.assign(self.g,1/std),tf.assign(self.b,-1.*mu/std)]

        require_init = tf.reduce_any(tf.is_nan(self.g))
        init_ops = tf.cond(require_init,_init,lambda : [self.g,self.b])

        with tf.control_dependencies(init_ops):

            w = tf.reshape(self.g,[1,1,tf.shape(self.v)[2],1]) * tf.nn.l2_normalize(self.v,axis=[0,1,3])
            return tf.nn.bias_add(
                tf.nn.conv2d_transpose(input_var,w,
                                       output_shape=shapes,
                                       strides=self.strides,
                                       padding='SAME',
                                       data_format='NHWC'),
                self.b,data_format='NHWC',name=name)

    def get_variables(self):
        #TODO: self.v should be l2-normalized or not? / currently not.
        return {'v':self.v,'b':self.b,'g':self.g}

class LayerNorm():
    def __init__(self,name,axis,out_dim=None,epsilon=1e-7,data_format='NHWC') :
        """
        out_dim: Recentering by adding bias again.
                 The previous bias can be ignored while normalization.
                 (when you normalize over channel only)
        """
        assert data_format=='NCHW' or data_format=='NHWC'
        assert len(axis) != 1 or (len(axis) == 1 and out_dim != None)

        """
        TODO: Track Moving mean and variance, and use this statistics.
        with tf.variable_scope(name):
            self.moving_mean = tf.get_variable('moving_mean',[dims], initializer=tf.constant_initializer(0.0), trainable=False)
            self.moving_variance = tf.get_variable('moving_variance',[dims], initializer=tf.constant_initializer(1.0), trainable=False)
        """

        if out_dim is not None:
            with tf.variable_scope(name) :
                self.gamma= tf.get_variable('gamma',[1,1,1,out_dim], initializer=tf.constant_initializer(1.0))
                self.beta = tf.get_variable('beta',[out_dim], initializer=tf.constant_initializer(0.0))
        else:
            self.gamma = None
            self.beta = None

        self.axis = axis
        self.epsilon = epsilon
        self.data_format = data_format
        self.name = name

    def __call__(self,input_var,**kwargs) :
        mean, var = tf.nn.moments(input_var, self.axis, keep_dims=True)
        ret = (input_var - mean) / tf.sqrt(var+self.epsilon)

        if self.gamma is None :
            return ret
        else:
            return tf.nn.bias_add(ret*self.gamma,
                                  self.beta,data_format=self.data_format)

    def get_variables(self):
        return {'gamma':self.gamma,'beta':self.beta} if self.gamma is not None else {}

class InstanceNorm():
    def __init__(self,name,format='NCHW',epsilon=1e-5) :
        assert(format=='NCHW' or format=='NHWC')
        self.axis = [2,3] if format == 'NCHW' else [1,2]

        self.epsilon = epsilon
        self.name = name

    def __call__(self,input_var) :
        mean, var = tf.nn.moments(input_var, self.axis, keep_dims=True)
        return (input_var - mean) / tf.sqrt(var+self.epsilon)

class BatchNorm(object):
    def __init__(self,name,dims,axis=1,epsilon=1e-3,momentum=0.999,center=True,scale=True) :
        self.momentum = momentum
        self.epsilon = epsilon
        self.axis = axis
        self.center=center
        self.scale=scale
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('bn') :
                self.gamma= tf.get_variable('gamma',[dims], initializer=tf.constant_initializer(1.0))
                self.beta = tf.get_variable('beta',[dims], initializer=tf.constant_initializer(0.0))
                self.moving_mean = tf.get_variable('moving_mean',[dims], initializer=tf.constant_initializer(0.0), trainable=False)
                self.moving_variance = tf.get_variable('moving_variance',[dims], initializer=tf.constant_initializer(1.0), trainable=False)
        self.scope = scope

    def __call__(self,input_var,is_training,**xargs) :
        with tf.variable_scope(self.scope) :
            return tf.layers.batch_normalization(
                input_var,
                axis=self.axis,
                momentum=self.momentum,
                epsilon=self.epsilon,
                center=self.center,
                scale=self.scale,
                training=is_training,
                reuse=True,
                name='bn')
        """
        ---Do NOT forget to add update_ops dependencies for your loss function.---
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,tf.get_default_graph().get_name_scope())
        #And, do not make any scope inside map_fn, since scope.name will not work...(it is corrupted by map_fn.)
        print(update_ops)
        with tf.control_dependencies(update_ops):
        """
    def get_variables(self):
        return {}

class Lrelu(object):
    def __init__(self,leak=0.2,name='lrelu') :
        self.leak = leak
        self.name = name
    def __call__(self, x, **kwargs) :
        return tf.maximum(x, self.leak*x, name=self.name)
    def get_variables(self):
        return {}

class ResidualBlock() :
    def __init__(self,name,filters,filter_size=3,non_linearity=Lrelu,normal_method=InstanceNorm) :
        self.conv_1 = Conv2d(name+'_1',filters,filters,filter_size,filter_size,1,1)
        self.normal = normal_method(name+'_norm')
        self.nl = non_linearity()
        self.conv_2 = Conv2d(name+'_2',filters,filters,filter_size,filter_size,1,1)
    def __call__(self,input_var) :
        _t = self.conv_1(input_var)
        _t = self.normal(_t)
        _t = self.nl(_t)
        _t = self.conv_2(_t)
        return input_var + _t
