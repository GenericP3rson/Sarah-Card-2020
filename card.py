import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import cv2

from IPython import display

timages = []
for filename in os.listdir(os.getcwd() + "/data"):
    print(os.getcwd() + "/data/" + filename)
    cap = cv2.imread(os.getcwd()  + "/data/" + filename)
    if cap.shape[0]!=0 and cap.shape[1]!=0:
        cap = cv2.resize(cap, (100, 100))
        timages.append(cap)
        print(filename)
timages = np.array(timages)
timages = (timages - 127.5) / 127.5

batch = 256
data = tf.data.Dataset.from_tensor_slices(timages).shuffle(60000).batch(batch)

def genemodel():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Dense(
        25*25*512, use_bias=False, input_shape=(100,)))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    m.add(tf.keras.layers.Reshape((25, 25, 512)))
    assert m.output_shape == (None, 25, 25, 512)

    m.add(tf.keras.layers.Conv2DTranspose(
        256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert m.output_shape == (None, 25, 25, 256)
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    m.add(tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert m.output_shape == (None, 50, 50, 128)
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    m.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert m.output_shape == (None, 100, 100, 3)

    return m


def discmodel():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[100, 100, 3]))
    m.add(tf.keras.layers.LeakyReLU())
    m.add(tf.keras.layers.Dropout(0.3))

    m.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    m.add(tf.keras.layers.LeakyReLU())

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(1))

    return m

generator = genemodel()
discriminator = discmodel()

generator.summary()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def dloss(output, foutput):
    loss = cross_entropy(tf.ones_like(output), output)
    floss = cross_entropy(tf.zeros_like(foutput), foutput)
    lloss = loss + floss
    return lloss

def gloss(foutput):
    return cross_entropy(tf.ones_like(foutput), foutput)


gopt = tf.keras.optimizers.Adam(1e-4)
dopt = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 15
seed = tf.random.normal([16, 100])

@tf.function
def train_step(images):
    noise = tf.random.normal([batch, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        geneimg = generator(noise, training=True)
        
        output = discriminator(images, training=True)
        # print("CHECK 1")
        foutput = discriminator(geneimg, training=True)
        
        gen_loss = gloss(foutput)
        disc_loss = dloss(output, foutput)

    gradG = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradD = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gopt.apply_gradients(zip(gradG, generator.trainable_variables))
    dopt.apply_gradients(zip(gradD, discriminator.trainable_variables))


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gopt,
                                 discriminator_optimizer=dopt,
                                 generator=generator,
                                 discriminator=discriminator)

def train(data, epochs):
    for epoch in range(epochs):
        start = time.time()

        for img in data:
            train_step(img)

        display.clear_output(wait=True)
        generate(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate(generator, epochs, seed)


def generate(model, epoch, test_input):
    pred = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(pred.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('sarahGENERATEDcards/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


train(data, EPOCHS)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
