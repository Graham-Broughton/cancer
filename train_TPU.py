import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_gcs_config

import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm.notebook import tqdm
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from keras_cv_attention_models import convnext

import os
import time
import pickle
import math
import random
import sys
import cv2
import gc
from datetime import datetime
import warnings; warnings.simplefilter('ignore')
import dotenv
dotenv.load_dotenv()

from config import CFG
CFG = CFG()
print(f'Tensorflow Version: {tf.__version__}')
print(f'Python Version: {sys.version}')

now = datetime.now().strftime("%d-%b-%Y %H-%M-%S")
np.save(now, np.array([now]))

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', TPU.master())
except ValueError:
    print('Running on GPU')
    TPU = None

if TPU:
    IS_TPU = True
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
else:
    IS_TPU = False
    STRATEGY = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

N_REPLICAS = STRATEGY.num_replicas_in_sync
print(f'N_REPLICAS: {N_REPLICAS}, IS_TPU: {IS_TPU}')

SEED = CFG.SEED
DEBUG = False

# Image dimensions
IMG_HEIGHT = CFG.IMG_HEIGHT
IMG_WIDTH = CFG.IMG_WIDTH
N_CHANNELS = CFG.N_CHANNELS
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
N_SAMPLES_TFRECORDS = CFG.N_SAMPLES_RECORD

# Peak Learning Rate
LR_MAX = CFG.LR_MAX * N_REPLICAS
WD_RATIO = CFG.WD_RATIO

N_WARMUP_EPOCHS = CFG.WARMUP_EPOCHS
N_EPOCHS = CFG.EPOCHS

# Batch size
BATCH_SIZE = CFG.BATCH_SIZE * N_REPLICAS

# Is Interactive Flag and COrresponding Verbosity Method
IS_INTERACTIVE = False
VERBOSE = 1 if IS_INTERACTIVE else 2

# Tensorflow AUTO flag
AUTO = tf.data.experimental.AUTOTUNE

print(f'BATCH_SIZE: {BATCH_SIZE}')

def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything()

# Train DataFrame
train = pd.read_csv('gs://kaggle-creds/train.csv')
test = pd.read_csv('gs://kaggle-creds/test.csv')
sample = pd.read_csv('gs://kaggle-creds/sample_submission.csv')

def tf_rand_int(minval, maxval, dtype=tf.int64):
    minval = tf.cast(minval, dtype)
    maxval = tf.cast(maxval, dtype)
    return tf.random.uniform(shape=(), minval=minval, maxval=maxval, dtype=dtype)

# chance of 1 in k
def one_in(k):
    return 0 == tf_rand_int(0, k)
# Function to benchmark the dataset
def benchmark_dataset(dataset, num_epochs=3, n_steps_per_epoch=10, bs=BATCH_SIZE):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for idx, (inputs, labels) in enumerate(dataset.take(n_steps_per_epoch + 1)):
            if idx == 0:
                epoch_start = time.perf_counter()
            elif idx == 1 and epoch_num == 0:
                image = inputs['image']
                print(f'image shape: {image.shape}, labels shape: {labels.shape}, image dtype: {image.dtype}, labels dtype: {labels.dtype}')
            else:
                pass
        
        epoch_t = time.perf_counter() - epoch_start
        mean_step_t = round(epoch_t / n_steps_per_epoch * 1000, 1)
        n_imgs_per_s = int(1 / (mean_step_t / 1000) * bs)
        print(f'epoch {epoch_num} took: {round(epoch_t, 2)} sec, mean step duration: {mean_step_t}ms, images/s: {n_imgs_per_s}')

# Decodes the TFRecords
def decode_image(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
    })
    
    # Decode PNG Image
    image = tf.io.decode_jpeg(features['image'], channels=N_CHANNELS)
    # Explicit reshape needed for TPU
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, N_CHANNELS])

    target = features['target']
    
    return { 'image': image }, target

def augment_image(X, y):
    image = X['image']
    
    # Random Brightness
    image = tf.image.random_brightness(image, 0.10)
    
    # Random Contrast
    image = tf.image.random_contrast(image, 0.90, 1.10)
    
    # Random JPEG Quality
    image = tf.image.random_jpeg_quality(image, 75, 100)
    
    # Random crop image with maximum of 10%
    ratio = tf.random.uniform([], 0.75, 1.00)
    img_height_crop = tf.cast(ratio * IMG_HEIGHT, tf.int32)
    img_width_crop = tf.cast(ratio * IMG_WIDTH, tf.int32)
    # Random offset for crop
    img_height_offset = tf_rand_int(0, IMG_HEIGHT - img_height_crop)
    img_width_offset = 0
    # Crop And Resize
    image = tf.slice(image, [img_height_offset, img_width_offset, 0], [img_height_crop, img_width_crop, N_CHANNELS])
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    # Clip pixel values in range [0,255] to prevent underflow/overflow
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    
    return { 'image': image }, y

# Undersample majority class (0/negative) by randomly dropping them
def undersample_majority(X, y):
    # Filter 2/3 of negative samples to upsample positive samples by a factor 3
    return y == 1 or tf.random.uniform([]) > 0.66

TFRECORDS_FILE_PATHS = sorted(tf.io.gfile.glob('gs://kaggle-creds/tfrecords/*.tfrecords'))
print(f'Found {len(TFRECORDS_FILE_PATHS)} TFRecords')
# Train Test Split
TFRECORDS_TRAIN, TFRECORDS_VAL = train_test_split(TFRECORDS_FILE_PATHS, train_size=0.80, random_state=SEED, shuffle=True)
print(f'# TFRECORDS_TRAIN: {len(TFRECORDS_TRAIN)}, # TFRECORDS_VAL: {len(TFRECORDS_VAL)}')

def get_dataset(tfrecords, bs=BATCH_SIZE, val=False, debug=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    
    # Initialize dataset with TFRecords
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO, compression_type='GZIP')
    
    # Decode mapping
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)

    if not val:
        dataset = dataset.filter(undersample_majority)
        dataset = dataset.map(augment_image, num_parallel_calls=AUTO)
        dataset = dataset.with_options(ignore_order)
        if not debug:
            dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()        

    dataset = dataset.batch(bs, drop_remainder=not val)
    dataset = dataset.prefetch(AUTO)
    
    return dataset

# Get Train/Validation datasets
train_dataset = get_dataset(TFRECORDS_TRAIN, val=False, debug=False)
val_dataset = get_dataset(TFRECORDS_VAL, val=True, debug=False)

TRAIN_STEPS_PER_EPOCH = len(TFRECORDS_TRAIN) * N_SAMPLES_TFRECORDS // BATCH_SIZE
VAL_STEPS_PER_EPOCH = len(TFRECORDS_VAL) * N_SAMPLES_TFRECORDS // BATCH_SIZE
print(f'TRAIN_STEPS_PER_EPOCH: {TRAIN_STEPS_PER_EPOCH}, VAL_STEPS_PER_EPOCH: {VAL_STEPS_PER_EPOCH}')

# Sanity check, image and label statistics
X_batch, y_batch = next(iter(get_dataset(TFRECORDS_TRAIN, val=False)))
image = X_batch['image'].numpy()
print(f'image shape: {image.shape}, y_batch shape: {y_batch.shape}')
print(f'image dtype: {image.dtype}, y_batch dtype: {y_batch.dtype}')
print(f'image min: {image.min():.2f}, max: {image.max():.2f}')

# Label Distribution Train With Undersampled Majority Class
N = BATCH_SIZE
train_labels = []
for _, labels in tqdm(get_dataset(TFRECORDS_TRAIN, val=False).take(N), total=N):
    train_labels += labels.numpy().tolist()

val_labels = []
for _, labels in tqdm(get_dataset(TFRECORDS_VAL, val=True), total=VAL_STEPS_PER_EPOCH):
    val_labels += labels.numpy().tolist()

class pF1(tf.keras.metrics.Metric):
    def __init__(self, name='pF1', **kwargs):
        super(pF1, self).__init__(name=name, **kwargs)
        self.tc = self.add_weight(name='tc', initializer='zeros')
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tc.assign_add(tf.cast(tf.reduce_sum(y_true), tf.float32))
        self.tp.assign_add(tf.cast(tf.reduce_sum((y_pred[y_true == 1])), tf.float32))
        self.fp.assign_add(tf.cast(tf.reduce_sum((y_pred[y_true == 0])), tf.float32))

    def result(self):
        if self.tc == 0 or (self.tp + self.fp) == 0:
            return 0.0
        else:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tc)
            return 2 * (precision * recall) / (precision + recall)

    def reset_state(self):
        self.tc.assign(0)
        self.tp.assign(0)
        self.fp.assign(0)


def normalize(image):
    # Repeat channels to create 3 channel images required by pretrained ConvNextV2 models
    image = tf.repeat(image, repeats=3, axis=3)
    # Cast to float 32
    image = tf.cast(image, tf.float32)
    # Normalize with respect to ImageNet mean/std
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')

    return image

def get_model():
    print(f'Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}')
    print(f'Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}')

    with STRATEGY.scope():
        seed_everything()
        image = tf.keras.layers.Input(shape=INPUT_SHAPE, name='image', dtype=tf.uint8)
        image_norm = normalize(image)
        x = convnext.ConvNeXtV2Tiny(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            pretrained='imagenet21k-ft1k',
            num_classes=0
        )(image_norm)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_MAX, decay=LR_MAX * WD_RATIO, epsilon=1e-6)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        metrics = [
            pF1(),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy(),
        ]

        model = tf.keras.Model(inputs=image, outputs=output)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

tf.keras.backend.clear_session()
# enable XLA optmizations
tf.config.optimizer.set_jit(True)

model = get_model()

# Learning rate scheduler with logaritmic warmup and cosine decay
def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    
    if current_step < num_warmup_steps:
        return lr_max * 0.10 ** (num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]

# Tensorflow Learning Rate Scheduler does not update weight decay, need to do it manually in a custom callback
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')

# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

# Train model on TPU!
history = model.fit(
        train_dataset,
        steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
        validation_data = val_dataset,
        epochs = N_EPOCHS,
        verbose = VERBOSE,
        callbacks = [
            lr_callback,
            WeightDecayCallback(),
        ],
        class_weight = {
            0: 1.0,
            1: 5.0,
        },
    )

# Save model weights for inference
model.save_weights('model.h5')
try:
    model.save_weights(f'gs://kaggle-creds/{now}/model.h5')
except:
    pass

# Get true labels and predictions for validation set
y_true_val = []
y_pred_val = []
for X_batch, y_batch in tqdm(get_dataset(TFRECORDS_VAL, val=True), total=VAL_STEPS_PER_EPOCH):
    y_true_val += y_batch.numpy().tolist()
    y_pred_val += model.predict_on_batch(X_batch).squeeze().tolist()

# source: https://www.kaggle.com/code/sohier/probabilistic-f-score
# Competition Leaderboard Metric
def pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

# Plot pF1 by threshold plot to find best threshold
pf1_by_threshold = []
thresholds = np.arange(0, 1.01, 0.01)
for t in tqdm(thresholds):
    # Compute pF1 for each threshold
    pf1_by_threshold.append(
        pfbeta(y_true_val, y_pred_val > t)
    )
    
plt.figure(figsize=(15,8))
plt.title('F1 By Threshold', size=24)
plt.plot(pf1_by_threshold, label='F1 Score')

# Best threshold and pF1 score
arg_max = np.argmax(pf1_by_threshold)
val_max = np.max(pf1_by_threshold)
threshold_best = thresholds[arg_max]
plt.scatter(arg_max, val_max, color='red', label=f'Best Threshold {threshold_best:.2f}, pF1 Score: {val_max:.2f}')

# Plot pF1 by Threshold
plt.xticks(np.arange(0, 110, 10), [f'{t:.2f}' for t in np.arange(0, 1.1, 0.1)])
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.xlabel('Threshold')
plt.ylabel('pF1 Score')
plt.legend(fontsize=12)
plt.grid()
plt.show()
plt.savefig(f'gs://kaggle-creds/{now}/pF1_by_threshold.png')

def plot_history_metric(metric, f_best=np.argmax, ylim=None, yscale=None, yticks=None):
    plt.figure(figsize=(20, 10))
    
    values = history.history[metric]
    N_EPOCHS = len(values)
    val = 'val' in ''.join(history.history.keys())
    # Epoch Ticks
    if N_EPOCHS <= 20:
        x = np.arange(1, N_EPOCHS + 1)
    else:
        x = [1, 5] + [10 + 5 * idx for idx in range((N_EPOCHS - 10) // 5 + 1)]

    x_ticks = np.arange(1, N_EPOCHS+1)

    # Validation
    if val:
        val_values = history.history[f'val_{metric}']
        val_argmin = f_best(val_values)
        plt.plot(x_ticks, val_values, label=f'val')

    # summarize history for accuracy
    plt.plot(x_ticks, values, label=f'train')
    argmin = f_best(values)
    plt.scatter(argmin + 1, values[argmin], color='red', s=75, marker='o', label=f'train_best')
    if val:
        plt.scatter(val_argmin + 1, val_values[val_argmin], color='purple', s=75, marker='o', label=f'val_best')

    plt.title(f'Model {metric}', fontsize=24, pad=10)
    plt.ylabel(metric, fontsize=20, labelpad=10)

    if ylim:
        plt.ylim(ylim)

    if yscale is not None:
        plt.yscale(yscale)
        
    if yticks is not None:
        plt.yticks(yticks, fontsize=16)

    plt.xlabel('epoch', fontsize=20, labelpad=10)        
    plt.tick_params(axis='x', labelsize=8)
    plt.xticks(x, fontsize=16) # set tick step to 1 and let x axis start at 1
    plt.yticks(fontsize=16)
    
    plt.legend(prop={'size': 10})
    plt.grid()
    plt.show()
    plt.savefig(f'gs://kaggle-creds/{now}/{metric}.png')

plot_history_metric('loss', f_best=np.argmin)
plot_history_metric('pF1', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
plot_history_metric('f1_score', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
plot_history_metric('precision', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
plot_history_metric('recall', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
plot_history_metric('auc', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
plot_history_metric('binary_accuracy', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))
