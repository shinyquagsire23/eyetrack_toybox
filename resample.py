import tensorflow as tf
import numpy as np

from globals import *

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    #print (img.shape)
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    img = tf.cast(img, 'float32')
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = (x * tf.cast(max_x-1, 'float32'))
    y = (y * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.math.floormod(x0, max_x)
    x1 = tf.math.floormod(x1, max_x)
    y0 = tf.math.floormod(y0, max_y)
    y1 = tf.math.floormod(y1, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def draw_patches(buf, grid_x_1, grid_y_1, grid_x_2, grid_y_2, grid_x_3, grid_y_3):
    img_height = buf.shape[0]
    img_width = buf.shape[1]

    for i in range(0, Globals.NETWORK_PATCH_SIZE):
        for j in range(0, Globals.NETWORK_PATCH_SIZE):
            x = int(grid_x_1[i,j] * img_width)
            y = int(grid_y_1[i,j] * img_height)

            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[y, x, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y, x-1, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y-1, x, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 0 or i == 0 or j == 47 or i == 47:
                    buf[y, x, 0] = 0
                    buf[y, x, 1] = 255.0
                    buf[y, x, 2] = 0

            x = int(grid_x_2[i,j] * img_width)
            y = int(grid_y_2[i,j] * img_height)
            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[y, x, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y, x-1, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y-1, x, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 47 or i == 0 or j == 47 or i == 47:
                    buf[y, x, 0] = 0
                    buf[y, x, 1] = 255.0
                    buf[y, x, 2] = 0

            x = int(grid_x_3[i,j] * img_width)
            y = int(grid_y_3[i,j] * img_height)
            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[y, x, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y, x-1, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[y-1, x, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 0 or i == 47 or j == 47 or i == 47:
                    buf[y, x, 0] = 0
                    buf[y, x, 1] = 255.0
                    buf[y, x, 2] = 0

def draw_patches_batched(buf, grid_x_1, grid_y_1, grid_x_2, grid_y_2, grid_x_3, grid_y_3):
    img_height = buf.shape[1]
    img_width = buf.shape[2]

    for i in range(0, Globals.NETWORK_PATCH_SIZE):
        for j in range(0, Globals.NETWORK_PATCH_SIZE):
            x = int(grid_x_1[0,i,j] * img_width)
            y = int(grid_y_1[0,i,j] * img_height)

            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[0, y, x, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y, x-1, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y-1, x, 0] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 0 or i == 0 or j == 47 or i == 47:
                    buf[0, y, x, 0] = 0
                    buf[0, y, x, 1] = 255.0
                    buf[0, y, x, 2] = 0

            x = int(grid_x_2[0,i,j] * img_width)
            y = int(grid_y_2[0,i,j] * img_height)
            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[0, y, x, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y, x-1, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y-1, x, 1] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 47 or i == 0 or j == 47 or i == 47:
                    buf[0, y, x, 0] = 0
                    buf[0, y, x, 1] = 255.0
                    buf[0, y, x, 2] = 0

            x = int(grid_x_3[0,i,j] * img_width)
            y = int(grid_y_3[0,i,j] * img_height)
            if x < img_width-1 and x >= 1 and y < img_height-1 and y >= 1:
                buf[0, y, x, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y, x-1, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                #buf[0, y-1, x, 2] = (float(j)/float(Globals.NETWORK_PATCH_SIZE)) * 255.0
                if j == 0 or i == 47 or j == 47 or i == 47:
                    buf[0, y, x, 0] = 0
                    buf[0, y, x, 1] = 255.0
                    buf[0, y, x, 2] = 0

def hexagonal_patches_tf(buf, hex_scale, hex_shape, hex_pos, hex_rot):
    img_shape = tf.shape(buf)
    batch_size = Globals.TRAIN_BATCH_SIZE#img_shape[0]

    img_height = buf.shape[1]
    img_width = buf.shape[2]

    hex_rot = tf.reshape(tf.cast(hex_rot, tf.float32), (-1, 1))
    hex_pos = tf.reshape(tf.cast(hex_pos, tf.float32), (-1, 2))
    hex_shape = tf.reshape(tf.cast(hex_shape, tf.float32), (-1, 1))
    hex_scale = tf.reshape(tf.cast(hex_scale, tf.float32), (-1, 1))

    one_px_x = (1.0/img_width)
    one_px_y = (1.0/img_height)

    all_shift_x = hex_pos[:, 0:1]# * one_px_x
    all_shift_y = hex_pos[:, 1:2]# * one_px_y

    rot_coef = tf.reshape(tf.stack([tf.experimental.numpy.cos(hex_rot), tf.experimental.numpy.sin(hex_rot), -tf.experimental.numpy.sin(hex_rot), tf.experimental.numpy.cos(hex_rot)]), (-1, 2, 2))

    hex_radius = float(Globals.NETWORK_PATCH_SIZE)
    hex_h = tf.sqrt(hex_radius**2-(hex_radius/2)**2)

    hex_vector_1_s = tf.ones((batch_size, 1))
    hex_vector_2_s = tf.ones((batch_size, 1))
    hex_vector_3_s = tf.ones((batch_size, 1))

    hex_vector_1_s *= hex_scale
    hex_vector_2_s *= hex_scale
    hex_vector_3_s *= hex_scale

    hex_vector_1 = tf.convert_to_tensor([[hex_radius*0.5, -hex_h]]) * hex_vector_1_s
    hex_vector_2 = tf.convert_to_tensor([[hex_radius*0.5, hex_h]]) * hex_vector_2_s
    hex_vector_3 = tf.convert_to_tensor([[-hex_radius, 0.0]]) * hex_vector_3_s

    hex_vector_1 = tf.keras.backend.batch_dot(hex_vector_1, rot_coef)
    hex_vector_2 = tf.keras.backend.batch_dot(hex_vector_2, rot_coef)
    hex_vector_3 = tf.keras.backend.batch_dot(hex_vector_3, rot_coef)

    hex_vector_1 = tf.reshape(hex_vector_1, (-1, 2))
    hex_vector_2 = tf.reshape(hex_vector_2, (-1, 2,))
    hex_vector_3 = tf.reshape(hex_vector_3, (-1, 2,))

    all_shift_x += hex_vector_3[:, 0:1] * one_px_x
    all_shift_y += hex_vector_1[:, 1:2]* one_px_y

    all_shift_x = tf.expand_dims(all_shift_x, axis=2)
    all_shift_y = tf.expand_dims(all_shift_y, axis=2)

    shift_x_1 = tf.cast(0.0*one_px_x, tf.float32)
    shift_x_2 = tf.cast(-(hex_vector_3[:, 0]*one_px_x), tf.float32)
    shift_x_3 = tf.cast(0.0*one_px_x, tf.float32)

    shift_y_1 = tf.cast(0.0*one_px_y, tf.float32)
    shift_y_2 = tf.cast(0.0*one_px_y, tf.float32)
    shift_y_3 = tf.cast(-(hex_vector_1[:, 1]*one_px_y), tf.float32)

    shift_x_1 = tf.reshape(shift_x_1, (-1, 1, 1))
    shift_x_2 = tf.reshape(shift_x_2, (-1, 1, 1))
    shift_x_3 = tf.reshape(shift_x_3, (-1, 1, 1))

    shift_y_1 = tf.reshape(shift_y_1, (-1, 1, 1))
    shift_y_2 = tf.reshape(shift_y_2, (-1, 1, 1))
    shift_y_3 = tf.reshape(shift_y_3, (-1, 1, 1))

    points_x_1 = tf.linspace([0.0], -(hex_vector_3[:, 0]*one_px_x), Globals.NETWORK_PATCH_SIZE)
    points_y_1 = tf.linspace([0.0], -(hex_vector_1[:, 1]*one_px_y), Globals.NETWORK_PATCH_SIZE)
    points_x_2 = tf.linspace([0.0], (hex_vector_2[:, 0] * one_px_x), Globals.NETWORK_PATCH_SIZE)
    points_y_2 = tf.linspace([0.0], -(hex_vector_1[:, 1]*one_px_y), Globals.NETWORK_PATCH_SIZE)
    points_x_3 = tf.linspace([0.0], -(hex_vector_3[:, 0]*one_px_x), Globals.NETWORK_PATCH_SIZE)
    points_y_3 = tf.linspace([0.0], (hex_vector_2[:, 1]*one_px_y), Globals.NETWORK_PATCH_SIZE)
    
    points_x_1 = tf.reshape(points_x_1, (-1, Globals.NETWORK_PATCH_SIZE, 1))
    points_y_1 = tf.reshape(points_y_1, (-1, Globals.NETWORK_PATCH_SIZE, 1))
    points_x_2 = tf.reshape(points_x_2, (-1, Globals.NETWORK_PATCH_SIZE, 1))
    points_y_2 = tf.reshape(points_y_2, (-1, Globals.NETWORK_PATCH_SIZE, 1))
    points_x_3 = tf.reshape(points_x_3, (-1, Globals.NETWORK_PATCH_SIZE, 1))
    points_y_3 = tf.reshape(points_y_3, (-1, Globals.NETWORK_PATCH_SIZE, 1))

    points_x_1 += shift_x_1
    points_y_1 += shift_y_1
    points_x_2 += shift_x_2
    points_y_2 += shift_y_2
    points_x_3 += shift_x_3
    points_y_3 += shift_y_3

    def fn_skew_x_1(i):
        shift = ((float(Globals.NETWORK_PATCH_SIZE) - i) * one_px_x) * (hex_vector_1[:, 0] / float(Globals.NETWORK_PATCH_SIZE))
        shift = tf.reshape(shift, (-1, 1))
        return tf.reshape(tf.linspace(shift, shift, Globals.NETWORK_PATCH_SIZE), (-1, Globals.NETWORK_PATCH_SIZE))

    def fn_skew_y_1(i):
        shift = (hex_vector_3[:, 1] * one_px_y)
        shift = tf.reshape(shift, (-1, 1))
        return tf.reshape(tf.linspace(shift, 0.0, Globals.NETWORK_PATCH_SIZE), (-1, Globals.NETWORK_PATCH_SIZE))

    def fn_skew_y_2(i):
        shift = (hex_vector_2[:, 1] * one_px_y) #(one_px_y * i) # 
        shift = tf.reshape(shift, (-1, 1))
        return tf.reshape(tf.linspace(0.0, shift, Globals.NETWORK_PATCH_SIZE), (-1, Globals.NETWORK_PATCH_SIZE))

    def fn_skew_x_3(i):
        shift = (one_px_x * i) * (hex_vector_2[:, 0] / float(Globals.NETWORK_PATCH_SIZE))
        shift = tf.reshape(shift, (-1, 1))
        return tf.reshape(tf.linspace(shift, shift, Globals.NETWORK_PATCH_SIZE), (-1, Globals.NETWORK_PATCH_SIZE))

    skew_shape = (-1, Globals.NETWORK_PATCH_SIZE,Globals.NETWORK_PATCH_SIZE)
    skew_x_1 = tf.reshape(tf.stack([fn_skew_x_1(i) for i in range(0, Globals.NETWORK_PATCH_SIZE)], axis=1), skew_shape)
    skew_y_1 = tf.reshape(tf.stack([fn_skew_y_1(i) for i in range(0, Globals.NETWORK_PATCH_SIZE)], axis=1), skew_shape)
    skew_y_2 = tf.reshape(tf.stack([fn_skew_y_2(i) for i in range(0, Globals.NETWORK_PATCH_SIZE)], axis=1), skew_shape)
    skew_x_3 = tf.reshape(tf.stack([fn_skew_x_3(i) for i in range(0, Globals.NETWORK_PATCH_SIZE)], axis=1), skew_shape)

    def gridify(t):
        return tf.meshgrid(t[0],t[1])

    grid_x_1, grid_y_1 = tf.map_fn(gridify, [points_x_1, points_y_1])
    grid_x_2, grid_y_2 = tf.map_fn(gridify, [points_x_2, points_y_2])
    grid_x_3, grid_y_3 = tf.map_fn(gridify, [points_x_3, points_y_3])

    grid_x_1 = tf.reshape(grid_x_1, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))
    grid_y_1 = tf.reshape(grid_y_1, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))
    grid_x_2 = tf.reshape(grid_x_2, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))
    grid_y_2 = tf.reshape(grid_y_2, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))
    grid_x_3 = tf.reshape(grid_x_3, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))
    grid_y_3 = tf.reshape(grid_y_3, (-1, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE))


    funky_coef_1 = hex_shape*tf.sqrt(Globals.NETWORK_PATCH_SIZE*hex_scale)
    funky_coef_2 = hex_shape*tf.sqrt(Globals.NETWORK_PATCH_SIZE*hex_scale)
    funky_coef_3 = hex_shape*tf.sqrt(Globals.NETWORK_PATCH_SIZE*hex_scale)
    

    grid_x_1 += skew_x_1
    grid_y_1 += skew_y_1

    grid_x_2 += skew_x_1
    grid_y_2 += skew_y_2

    grid_x_3 += skew_x_3
    grid_y_3 += skew_y_1

    grid_x_1 += all_shift_x
    grid_x_2 += all_shift_x
    grid_x_3 += all_shift_x

    grid_y_1 += all_shift_y
    grid_y_2 += all_shift_y
    grid_y_3 += all_shift_y


    # Arc the edges inwards into a star/outwards into a circle
    # I got all of this via trial and error, do not ask me how it works
    sin_space = tf.experimental.numpy.sin(np.pi * np.linspace(0.0, 1.0, Globals.NETWORK_PATCH_SIZE, dtype=np.float32))
    linear_0_1 = np.reshape(np.linspace(0.0, 1.0, Globals.NETWORK_PATCH_SIZE, dtype=np.float32), (1, 1, Globals.NETWORK_PATCH_SIZE))

    sin_space = tf.reshape(sin_space, (1, Globals.NETWORK_PATCH_SIZE, 1))
    curve_grid_i_invj = sin_space * (1.0 - linear_0_1)
    curve_grid_i_j = sin_space * linear_0_1
    curve_grid_j_invi = tf.transpose(curve_grid_i_invj, [0, 2, 1])
    curve_grid_j_i = tf.transpose(curve_grid_i_j, [0, 2, 1])

    curve_grid_i_invj = tf.broadcast_to(curve_grid_i_invj, (batch_size, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE,))
    curve_grid_i_j = tf.broadcast_to(curve_grid_i_j, (batch_size, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE,))
    curve_grid_j_invi = tf.broadcast_to(curve_grid_j_invi, (batch_size, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE,))
    curve_grid_j_i = tf.broadcast_to(curve_grid_j_i, (batch_size, Globals.NETWORK_PATCH_SIZE, Globals.NETWORK_PATCH_SIZE,))


    def batch_mul(t):
        a = t[0]
        b = t[1]
        c = t[2]
        out = a*b*c
        return [out, out, out]


    grid_x_1 -= tf.map_fn(batch_mul, [curve_grid_i_invj * (one_px_x), funky_coef_1, tf.experimental.numpy.cos(hex_rot)])[0]
    grid_x_1 += tf.map_fn(batch_mul, [curve_grid_j_invi * (one_px_x), funky_coef_1, tf.experimental.numpy.sin(hex_rot)])[0]

    grid_y_1 -= tf.map_fn(batch_mul, [curve_grid_j_invi * (one_px_y), funky_coef_1, tf.experimental.numpy.cos(hex_rot)])[0]
    grid_y_1 -= tf.map_fn(batch_mul, [curve_grid_i_invj * (one_px_y), funky_coef_1, tf.experimental.numpy.sin(hex_rot)])[0]

    grid_x_2 += tf.map_fn(batch_mul, [curve_grid_i_j * (one_px_x), funky_coef_2, tf.experimental.numpy.cos(hex_rot + np.pi/6)])[0]
    grid_x_2 += tf.map_fn(batch_mul, [curve_grid_j_invi * (one_px_x), funky_coef_2, tf.experimental.numpy.sin(hex_rot + np.pi/6)])[0]

    grid_y_2 -= tf.map_fn(batch_mul, [curve_grid_j_invi * (one_px_y), funky_coef_2, tf.experimental.numpy.cos(hex_rot + np.pi/6)])[0]
    grid_y_2 += tf.map_fn(batch_mul, [curve_grid_i_j * (one_px_y), funky_coef_2, tf.experimental.numpy.sin(hex_rot + np.pi/6)])[0]

    grid_x_3 -= tf.map_fn(batch_mul, [curve_grid_i_invj * (one_px_x), funky_coef_3, tf.experimental.numpy.cos(hex_rot)])[0]
    grid_x_3 -= tf.map_fn(batch_mul, [curve_grid_j_i * (one_px_x), funky_coef_3, tf.experimental.numpy.sin(hex_rot)])[0]

    grid_y_3 += tf.map_fn(batch_mul, [curve_grid_j_i * (one_px_y), funky_coef_3, tf.experimental.numpy.cos(hex_rot)])[0]
    grid_y_3 -= tf.map_fn(batch_mul, [curve_grid_i_invj * (one_px_y), funky_coef_3, tf.experimental.numpy.sin(hex_rot)])[0]
    
    # Finally, do the sampling
    cropped_frame_1 = bilinear_sampler(buf, grid_x_1, grid_y_1)
    cropped_frame_2 = bilinear_sampler(buf, grid_x_2, grid_y_2)
    cropped_frame_3 = bilinear_sampler(buf, grid_x_3, grid_y_3)

    return [cropped_frame_1, grid_x_1, grid_y_1], [cropped_frame_2, grid_x_2, grid_y_2], [cropped_frame_3, grid_x_3, grid_y_3]

def hexagonal_patches_simple_tf(buf, hex_scale, hex_shape, hex_pos, hex_rot):
    return hexagonal_patches_tf(buf, hex_scale, hex_shape, hex_pos, hex_rot)