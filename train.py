
import os

import tensorflow as tf
from tensorflow import keras
import preprocessing
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from typing import Union, Callable, List
from tensorflow.python.keras.losses import LossFunctionWrapper
from typeguard import typechecked
from typing import Optional, Union, Callable

print(tf.version.VERSION)


def parse_img(img, label):
    return (img, label)
    
def create_dataset(filenames, labels, is_training=True):
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #for element in dataset: 
     #     print(element) 
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_img, num_parallel_calls=autotune)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(batch_size)
    # Fetch batches in the background while |the model is training.
    dataset = dataset.prefetch(buffer_size=autotune)
    
    return datasettrain_dataset = parse_img(images_train, labels_train)
    
def model_net(mo):
    if mo=='mobile_net':
        base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),
                                               include_top=False,
                                               weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(776, activation=None), # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

        ])
    if mo =='vgg_net':
        inputShape = (32, 32, 3)
        chanDim = -1
    # build the model using Keras' Sequential API
        model = tf.keras.Sequential([
        # CONV => RELU => BN => POOL layer set
            Conv2D(16, (3, 3), padding="same", input_shape=inputShape),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            MaxPooling2D(pool_size=(2, 2)),
        # (CONV => RELU => BN) * 2 => POOL layer set
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            MaxPooling2D(pool_size=(2, 2)),
        # (CONV => RELU => BN) * 3 => POOL layer set
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            BatchNormalization(axis=chanDim),
            MaxPooling2D(pool_size=(2, 2)),
        # first (and only) set of FC => RELU layers
            Flatten(),
            Dense(776),
            Activation("relu"),
            BatchNormalization(),
            Dropout(0.5),
        ])
    
    if mo=='resnet':
        base_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3),
                                               include_top=False,
                                               weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(776, activation=None), # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

    ])
    return model

@tf.function
def pairwise_distance(feature: TensorLike, squared: bool = False):
    
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
    )

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data])
    )
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def _masked_minimum(data, mask, dim=1):
    
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums

def _masked_maximum(data, mask, dim=1):
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums

def triplet_semihard_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:


    labels, embeddings = y_true, y_pred

    convert_to_float32 = (embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16)
    precise_embeddings = ( tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings)

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix
    pdist_matrix = pairwise_distance(precise_embeddings, squared=True)


    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask.
    pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
    mask = tf.math.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        tf.math.greater(
            pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
        ),
    )
    mask_final = tf.reshape(
        tf.math.greater(
            tf.math.reduce_sum(
                tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
            ),
            0.0,
        ),
        [batch_size, batch_size],
    )
    mask_final = tf.transpose(mask_final)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    mask = tf.cast(mask, dtype=tf.dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.math.reduce_sum(mask_positives)

    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
        ),
        num_positives,
    )

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss

#helper function to plot image
def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(data[idxs[i],:,:,:])
        ax.axis('off')
    plt.show()

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)
def parse_img(img, label):
    return (img, label)


# Find k nearest neighbour using cosine similarity
def find_k_nn(normalized_train_vectors,vec,k):
    dist_arr = np.matmul(normalized_train_vectors, tf.transpose(vec))
    return np.argsort(-dist_arr.flatten())[:k]

def MeanAveragePrecision2(indx_list,test_image_indexes, train_images, test_images):
    mAP=0
    for i in range(len(indx_list)):
        #print(test_image_indexes[i],test_label[test_image_indexes[i]])
        Qlabel=labels_test[test_image_indexes[i]]
        #print("getting train images")
        c=1
        AP=0
        for j in range(len(indx_list[0])):
         #   print(indx_list[i][j],train_label[indx_list[i][j]])
            dblabel=labels_train[indx_list[i][j]]
            if(dblabel==Qlabel):
                AP=AP+c/(j+1)
          #      print("internal",AP)
                c=c+1
            elif(dblabel!=Qlabel):
                AP=AP+0/(j+1)
        #print(c)
        AP=AP/c
        print(AP)
        mAP=mAP+AP
    mAP=mAP/len(indx_list)
    print(mAP)
    return(mAP)
    
class TripletSemiHardLoss(LossFunctionWrapper):
   
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            triplet_semihard_loss,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
            distance_metric=distance_metric,
        )
        
def main():
    train = 'D:/veri-split/train/'
    test = 'D:/veri-split/test/'
    with tf.device('/GPU:0'):
        images_train, labels_train, images_val, labels_val, unique_train_label, map_train_indices = preprocessing.main_train(train)
        images_test, labels_test, unique_test_label, map_test_indices = preprocessing.main_test(test)

    val_dataset = parse_img(images_val, labels_val)
    test_dataset = parse_img(images_test, labels_test)

    autotune = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = 1024
    batch_size = 512

    train_ds = create_dataset(train_dataset[0], train_dataset[1])
    val_ds = create_dataset(val_dataset[0], val_dataset[1])
    test_ds = create_dataset(test_dataset[0], test_dataset[1])


    Number = Union[
        float,
        int,
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]

    TensorLike = Union[
        List[Union[Number, list]],
        tuple,
        Number,
        np.ndarray,
        tf.Tensor,
        tf.SparseTensor,
        tf.Variable,
    ]

    FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
    AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
    mo = 'mobile_net'
    model = model_net(mo)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                    loss=TripletSemiHardLoss(),
                    metrics=['accuracy'])
    model.summary()
    checkpoint_path = mo+"/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    with tf.device('/GPU0:'):

        # Train the model with the new callback
        mobile.fit(train_ds,  
                  epochs=10,
                  validation_data = (val_ds),
                  callbacks=[cp_callback])  # Pass callback to training
    model = model_net(mo)

    # Evaluate the model
    loss, acc = model.evaluate(test_ds, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss,acc = model.evaluate(test_ds, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
    im = images_test[idx]
    idx = np.random.randint(0, len(images_test))
    label1=labels_test[idx]
    print(label1)
    #show the test image
    print("********** QUERY IMAGE **********")
    show_image(idx, images_test)
    checkpoint_path = mo+"/cp.ckpt"
    model = model_net(mo)
    model.load_weights(checkpoint_path)
    # Compute Vector representation for each training images and normalize those
    with tf.device('/GPU0:'):
        train_vectors = model(images_train)     
        normalized_train_vectors = train_vectors/np.linalg.norm(train_vectors,axis=1).reshape(-1,1)
        K = 20
        N = 20
        indx_list = []
        test_image_indexes = []
        _test_images = []
        for i in range(N):
            idx = i
            test_image_indexes.append(idx)
            _test_images.append(images_test[idx])
            #run the test image through the network to get the test features
        t = np.array(_test_images)
        search_vectors = model(t)
    normalized_search_vecs = search_vectors/np.linalg.norm(search_vectors,axis=1).reshape(-1,1)
    with tf.device('/GPU:0'):
        for i in range(len(normalized_search_vecs)):
            candidate_index = find_k_nn(normalized_train_vectors, normalized_search_vecs[i], K)
            indx_list.append(candidate_index)

        MeanAveragePrecision2(indx_list,test_image_indexes, images_train, images_test)
        
if __name__ == '__main__':
    main()
