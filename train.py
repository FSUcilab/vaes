"""Variational Auto-encoder

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import time
import datetime
import inspect
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import control_flow_ops
from models import *
from reconstructions import *
from loss import *
from nn_utils import whiten
from datasets import binarized_mnist
import argparse

def train(
        image_width,
        dim_x,
        dim_z,
        #encoder_net,
        #decoder_net,
        encoder_type,
        decoder,
        dataset,
        learning_rate=0.0001,
        optimizer=tf.train.AdamOptimizer,
        loss=elbo_loss,
        batch_size=100,
        results_dir='results',
        max_epochs=10,
        n_view=10,
        results_file=None,
        bn=False,
        **kwargs
        ):
    anneal_lr = kwargs.pop('anneal_lr', False)
    global_step = tf.Variable(0, trainable=False) # for checkpoint saving
    on_epoch = tf.placeholder(tf.float32, name='on_epoch')
    dt = datetime.datetime.now()
    results_file = results_file if results_file is not None else '/{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    results_dir += results_file
    os.mkdir(results_dir)

    # Get all the settings and save them.
    with open(results_dir + '/settings.txt', 'w') as f:
        args = inspect.getargspec(train).args
        settings = [locals()[arg] for arg in args]
        for s, arg in zip(settings, args):
            setting = '{}: {}'.format(arg, s)
            f.write('{}\n'.format(setting))
            print(setting)
        settings = locals()[inspect.getargspec(train).keywords]
        for kw, val in settings.items():
            setting = '{}: {}'.format(kw, val)
            f.write('{}\n'.format(setting))
            print(setting)

    # Make the neural neural_networks
    is_training = tf.placeholder(tf.bool)
    if bn:
        encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh, is_training=is_training)
    else: encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh, is_training=None)
    #decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
    encoder = encoder_type(encoder_net, dim_z, flow)

    # Build computation graph and operations
    x = tf.placeholder(tf.float32, [None, dim_x], 'x')
    x_w = tf.placeholder(tf.float32, [None, dim_x], 'x_w')
    e = tf.placeholder(tf.float32, (None, dim_z), 'noise')

    z_params, z = encoder(x_w, e)
    x_pred = decoder(z)

    kl_weighting = 1.0 - tf.exp(-on_epoch / kl_annealing_rate) if kl_annealing_rate is not None else 1
    monitor_functions = loss(x_pred, x, kl_weighting=kl_weighting, **z_params)
    train_loss, valid_loss = monitor_functions['train_loss'], monitor_functions['valid_loss']
    out_op = x_pred

    # Batch normalization stuff
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        train_loss = control_flow_ops.with_dependencies([updates], train_loss)

    # Optimizer with gradient clipping
    lr = tf.Variable(learning_rate)
    optimizer = optimizer(lr)
    gvs = optimizer.compute_gradients(train_loss)
    capped_gvs = [(tf.clip_by_norm(grad, 1), var) if grad is not None else (grad, var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # Make training and validation sets
    training_data, validation_data = dataset['train'], dataset['valid']
    n_train_batches, n_valid_batches = training_data.images.shape[0] / batch_size, validation_data.images.shape[0] / batch_size,
    print 'Loaded training and validation data'
    visualized, e_visualized = validation_data.images[:n_view], np.random.normal(0, 1, (n_view, dim_z))

    # Make summaries
    rec_summary = tf.image_summary("rec", vec2im(out_op, batch_size, image_width), max_images=10)
    for fn_name, fn in monitor_functions.items():
        tf.scalar_summary(fn_name, fn)
    summary_op = tf.merge_all_summaries()

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(results_dir, sess.graph)
    samples_list = []
    batch_counter = 0
    best_validation_loss = 1e100
    number_of_validation_failures = 0
    feed_dict = {}
    validation_losses, training_losses = [], []
    for epoch in range(max_epochs):
        feed_dict[on_epoch] = epoch
        start_time = time.time()
        l_t = 0
        for _ in xrange(n_train_batches):
            batch_counter += 1
            feed_dict[x], feed_dict[x_w] = training_data.next_batch(batch_size, whitened=False)
            feed_dict[e] = np.random.normal(0, 1, (batch_size, dim_z))
            feed_dict[is_training] = True
            _, l = sess.run([train_op, train_loss], feed_dict=feed_dict)
            if batch_counter % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, batch_counter)

            # Save the model checkpoint periodically.
            if batch_counter % 1000 == 0 or epoch == max_epochs:
                checkpoint_path = os.path.join(results_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
            l_t += l
        l_t /= n_train_batches
        training_losses.append(l_t)
        l_v = 0
        for _ in range(n_valid_batches):

            feed_dict[x], feed_dict[x_w] = validation_data.next_batch(batch_size, whitened=False)
            feed_dict[e] = np.random.normal(0, 1, (batch_size, dim_z))
            feed_dict[is_training] = False
            l_v_batched = sess.run(valid_loss, feed_dict=feed_dict)
            l_v += l_v_batched
        l_v /= n_valid_batches
        validation_losses.append(l_v)
        duration = time.time() - start_time
        examples_per_sec = (n_valid_batches + n_train_batches) * batch_size * 1.0 / duration
        print('Epoch: {:d}\t Weighted training loss: {:.2f}, Validation loss {:.2f} ({:.1f} examples/sec, {:.1f} sec/epoch)'.format(epoch, l, l_v, examples_per_sec, duration))

        if l_v > best_validation_loss:
            number_of_validation_failures += 1
        else:
            best_validation_loss = l_v
            number_of_validation_failures = 0

        if number_of_validation_failures == 5 and anneal_lr:
            lr /= 2
            learning_rate /= 2
            print "Annealing learning rate to {}".format(learning_rate)
            number_of_validation_failures = 0

        samples = sess.run([out_op], feed_dict={x: visualized, x_w: whiten(visualized), e: e_visualized, is_training: False})
        samples = np.reshape(samples, (n_view, image_width, image_width))
        samples_list.append(samples)
        #show_samples(samples, image_width

    np.save(results_dir + '/validation_losses.npy', validation_losses)
    np.save(results_dir + '/training_losses.npy', training_losses)
    np.save(results_dir + '/sample_visualizations.npy', np.array(samples_list))
    np.save(results_dir + '/real_visualizations.npy', np.reshape(visualized, (n_view,image_width, image_width)))

    visualize = False
    if visualize:
        for samples in samples_list:
            together = np.hstack((np.reshape(visualized, (n_view,image_width, image_width)), samples > 0.5))
            plot_images_together(together)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--basic', action='store_true')
    group.add_argument('--nf', action='store_true')
    group.add_argument('--iaf', action='store_true')
    group.add_argument('--hf', action='store_true')
    group.add_argument('--liaf', action='store_true')

    parser.add_argument('--anneal-lr', action='store_true')
    parser.add_argument('--flow', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Results file
    dt = datetime.datetime.now()
    results_file = '/{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)

    ### TRAINING SETTINGS
    dim_x, dim_z, enc_dims, dec_dims = 784, 40, [300, 300], [300, 300]
    decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
    flow = args.flow
    bn = True

    ### ENCODER
    if args.basic:
        #encoder_type = basic_encoder(encoder_net, dim_z)
        encoder_type = basic_encoder
        results_file += '-basic'
    if args.nf:
        #encoder_type = nf_encoder(encoder_net, dim_z, flow)
        encoder_type = nf_encoder
        results_file += '-NF-{}'.format(flow)
    if args.iaf:
        #encoder_type = iaf_encoder(encoder_net, dim_z, flow)
        encoder_type = iaf_encoder
        results_file += '-IAF-{}'.format(flow)
    if args.hf:
        #encoder_type = hf_encoder(encoder_net, dim_z, flow)
        encoder_type = hf_encoder
        results_file += '-HF-{}'.format(flow)
    if args.liaf:
        encoder_type = linear_iaf_encoder
        results_file += '-linIAF'
    ### DECODER
    decoder = basic_decoder(decoder_net, dim_x)

    ##############

    #kl_annealing_rate = 10
    kl_annealing_rate = None
    ### ENCODER
    #encoder, model_type = basic_encoder(encoder_net, dim_z), 'Vanilla VAE'
    #encoder, model_type = nf_encoder(encoder_net, dim_z, flow), 'Normalizing Flow'
    #encoder, model_type = hf_encoder(encoder_net, dim_z, flow), 'Householder Flow'
    #encoder, model_type = iaf_encoder(encoder_net, dim_z, flow), 'Inverse Autoregressive Flow'


    extra_settings = {
        # 'model_type':model_type,
        'flow': flow,
        'kl annealing rate':kl_annealing_rate,
        'anneal_lr': args.anneal_lr,
        'bn':bn,
        'enc_dims':enc_dims

    }

    #######################################
    ## TRAINING
    #######################################
    train(
    image_width=28,
    dim_x=dim_x,
    dim_z=dim_z,
    #decoder_net=decoder_net,
    encoder_type=encoder_type,
    decoder=decoder,
    dataset=binarized_mnist(),
    learning_rate=0.0002,
    optimizer=tf.train.AdamOptimizer,
    loss=elbo_loss,
    batch_size=100,

    results_dir='results',
    results_file=results_file,
    max_epochs=200,
    **extra_settings
        )