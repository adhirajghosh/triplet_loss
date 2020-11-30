import tensorflow as tf
import preprocessing
import model
import os

batch_size = 512 
train_iter = 1500
step = 50
learning_rate = 0.01
momentum = 0.99
model_name = 'conv_net'


images_train, labels_train, images_test, labels_test, unique_train_label, map_train_indices = preprocessing.main()
shape = [None]+ list(images_train.shape[1:])
print("shape is", shape)
global_step = tf.Variable(0, trainable=False)

checkpoint_dir = "checkpoint/"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
checkpoint =  tf.train.Checkpoint(optimizer = opt)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

loss_fn = lambda: model.triplet_loss(anchor_o, positive_o, negative_o, 0.3)

with tf.device('/GPU:0'):

    for i in range(train_iter):
        batch_a, batch_p, batch_n = preprocessing.get_triplets_batch(batch_size, images_train, unique_train_label, map_train_indices)
        anchor_o = model.conv_net(batch_a)
        positive_o = model.conv_net(batch_p)
        negative_o = model.conv_net(batch_n)
        var_list = [anchor_o, positive_o, negative_o]
        opt.minimize(loss_fn, var_list)

    status.assert_consumed()
    checkpoint.save(file_prefix=checkpoint_prefix) 

print("training completed succesfully")