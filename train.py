
import tensorflow as tf
import matplotlib.pyplot as plt
#from model import dpdnet 

IMAGE_SIZE = [212,256]
BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE

image_feature_description = {

    'image_input': tf.io.FixedLenFeature([], tf.string),
    'image_output': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image_inp = decode_image(parsed_example['image_input'])
  image_out = decode_image(parsed_example['image_output'])
    
  return image_inp,image_out


def decode_image(image_data):
    #decode image 
    image = tf.image.decode_jpeg(image_data, channels=1)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,IMAGE_SIZE)
    return image

data_dir = GCS_DS_PATH + '/*.tfrecord'

def get_dataset(data_dir):
    filenames = tf.io.gfile.glob(data_dir)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(1000).batch(BATCH_SIZE)
    return dataset


#define optimizer,loss_fn and model 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
model = dpdnet()

epochs = 5
train_iterator = iter(train_dataset)
for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x, y) in enumerate(train_iterator):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            output1,output2 = model(x, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value1 = loss_fn(y,output1)
            loss_value2 = loss_fn(y,output2)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value2, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 20 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value2))
            )
            print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))