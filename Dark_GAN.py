#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Input, concatenate, PReLU, Add
from tensorflow.keras import Model
#from IPython import display
import os
from PIL import Image
import time
import numpy as np
#tf.keras.backend.set_image_data_format('channels_first')

path_inp = 'F:/trial/'
checkpoint_dir = 'D:/Prasen/Pix2Pix/Checkpoints_Gen2'
reload_dir = 'D:/Prasen/Pix2Pix/IMP_Chechpoints/Exellents'
res_dire_1 = 'D:/Prasen/Pix2Pix/Results/Gen_1/'
res_dire_2 = 'D:/Prasen/Pix2Pix/Results/Gen_C/'

def norm_train(inp):
    image = tf.io.read_file(inp)    # Read any file and convert it into tensor
    
    image = tf.image.decode_jpeg(image)    # Convert tensor into uint8 type tensor such as image matrix

    
    w = tf.shape(image)[1]        # w is that coloumn value which will slpit a combined image into 2         

    w = w // 2
    
    real_image = image[:, :w, :]       # separating 2 images by taking half of a combined image    
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)  # Converting uint8 to float32
    real_image = tf.cast(real_image, tf.float32)
    
    input_image = (input_image / 255)
    real_image = (real_image / 255)


    return input_image, real_image


    

def norm_test(inp):
    image = tf.io.read_file(inp)    # Read any file and convert it into tensor
    
    image = tf.image.decode_jpeg(image)    # Convert tensor into uint8 type tensor such as image matrix

    input_image = tf.cast(image, tf.float32)  # Converting uint8 to float32
    
    input_image = (input_image / 255)
    
    return input_image

input_data = tf.data.Dataset.list_files(os.path.join(path_inp + 'train/*.jpg'))
input_data = input_data.map(norm_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
input_data = input_data.shuffle(200)
input_data = input_data.batch(1)

#if tf.keras.backend.image_data_format() == 'channels_first':
#        input_image = tf.transpose(input_image, [2, 0, 1])
#        real_image = tf.transpose(real_image, [2, 0, 1])

test_data = tf.data.Dataset.list_files(os.path.join(path_inp + 'test/*.jpg'))
test_data = test_data.map(norm_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(1)

#if tf.keras.backend.image_data_format() == 'channels_first':
#        input_image = tf.transpose(input_image, [2, 0, 1])
        
        
init = tf.random_normal_initializer(0.,0.02)

def Generator_model():
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        I = tf.keras.layers.Input(shape=[3,256,256])
    else:
        I = tf.keras.layers.Input(shape=[256,256,3])
    
    #I = Input(shape=[256,256,3])
    
    C1 = Conv2D(64, (9,9), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(I)
    L1 = LeakyReLU()(C1)
    
    C2 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L1)
    B2 = BatchNormalization(axis=1)(C2)
    L2 = LeakyReLU()(B2)
    
    C3 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L2)
    B3 = BatchNormalization(axis=1)(C3)
    L3 = LeakyReLU()(B3)
    
    A1 = Add()([L1,L3])
    
    C4 = Conv2D(64, (3,3),padding='same',  data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(A1)
    B4 = BatchNormalization(axis=1)(C4)
    L4 = LeakyReLU()(B4)
    
    C5 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L4)
    B5 = BatchNormalization(axis=1)(C5)
    L5 = LeakyReLU()(B5)
    
    U6 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L5)
    B6 = BatchNormalization(axis=1)(U6)
    L6 = LeakyReLU()(B6)
    
    A2 = Add()([L4, L6])
    
    U7 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(),kernel_initializer=init,use_bias=True)(A2)
    B7 = BatchNormalization(axis=1)(U7)
    L7 = LeakyReLU()(B7)
    
    U8 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(),kernel_initializer=init,use_bias=True)(L7)
    B8 = BatchNormalization(axis=1)(U8)
    L8 = LeakyReLU()(B8)
    
    U9 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L8)
    B9 = BatchNormalization(axis=1)(U9)
    L9 = LeakyReLU()(B9)
    
    A3 = Add()([L7, L9])
    
    U10 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A3)
    B10 = BatchNormalization(axis=1)(U10)
    L10 = LeakyReLU()(B10)
    
    U11 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L10)
    B11 = BatchNormalization(axis=1)(U11)
    L11 = LeakyReLU()(B11)
    
    U12 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L11)
    B12 = BatchNormalization(axis=1)(U12)
    L12 = LeakyReLU()(B12)
    
    A4 = Add()([L10, L12])
    
    U13 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A4)
    B13 = BatchNormalization(axis=1)(U13)
    L13 = LeakyReLU()(B13)
    
    U14 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L13)
    B14 = BatchNormalization(axis=1)(U14)
    L14 = LeakyReLU()(B14)
    
    U15 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L14)
    B15 = BatchNormalization(axis=1)(U15)
    L15 = LeakyReLU()(B15)
    
    A5 = Add()([L13, L15])
    
    U16 = Conv2D(256, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A5)
    B16 = BatchNormalization(axis=1)(U16)
    L16 = LeakyReLU()(B16)
        
    out = Conv2D(3, (1,1), kernel_initializer=init, data_format=tf.keras.backend.image_data_format(),  activation='tanh', use_bias=True)(L16)
    
    model = Model(inputs=I, outputs=out)
    
    return model


generator = Generator_model()
#generator.summary()

def Gen_2():
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        I = tf.keras.layers.Input(shape=[3,256,256])
    else:
        I = tf.keras.layers.Input(shape=[256,256,3])
        
    #I = Input(shape=[256,256,3])
    
    C1 = Conv2D(64, (9,9), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(I)
    L1 = LeakyReLU()(C1)
    
    C2 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L1)
    B2 = BatchNormalization(axis=1)(C2)
    L2 = LeakyReLU()(B2)
    
    C3 = Conv2D(64, (3,3), padding='same',  data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L2)
    B3 = BatchNormalization(axis=1)(C3)
    L3 = LeakyReLU()(B3)
    
    A1 = Add()([L1,L3])
    
    C4 = Conv2D(64, (3,3),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(A1)
    B4 = BatchNormalization(axis=1)(C4)
    L4 = LeakyReLU()(B4)
    
    C5 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L4)
    B5 = BatchNormalization(axis=1)(C5)
    L5 = LeakyReLU()(B5)
    
    U6 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L5)
    B6 = BatchNormalization(axis=1)(U6)
    L6 = LeakyReLU()(B6)
    
    A2 = Add()([L4, L6])
    
    U7 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A2)
    B7 = BatchNormalization(axis=1)(U7)
    L7 = LeakyReLU()(B7)
    
    U8 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(),kernel_initializer=init,use_bias=True)(L7)
    B8 = BatchNormalization(axis=1)(U8)
    L8 = LeakyReLU()(B8)
    
    U9 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L8)
    B9 = BatchNormalization(axis=1)(U9)
    L9 = LeakyReLU()(B9)
    
    A3 = Add()([L7, L9])
    
    U10 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A3)
    B10 = BatchNormalization(axis=1)(U10)
    L10 = LeakyReLU()(B10)
    
    U11 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L10)
    B11 = BatchNormalization(axis=1)(U11)
    L11 = LeakyReLU()(B11)
    
    U12 = Conv2D(64, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(L11)
    B12 = BatchNormalization(axis=1)(U12)
    L12 = LeakyReLU()(B12)
    
    A4 = Add()([L10, L12])
    
    U13 = Conv2D(128, (3,3), padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init,use_bias=True)(A4)
    B13 = BatchNormalization(axis=1)(U13)
    L13 = LeakyReLU()(B13)
        
    out = Conv2D(3, (1,1), kernel_initializer=init, data_format=tf.keras.backend.image_data_format(), activation='tanh', use_bias=True)(L13)
    
    model = Model(inputs=I, outputs=out)
    
    return model


generator_2 = Gen_2()
#generator_2.summary()

def Discriminator_model():
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        inp = tf.keras.layers.Input(shape=[3, 256, 256], name='input_image')
        tar = tf.keras.layers.Input(shape=[3, 256, 256], name='target_image')
    else:
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    
    
    #inp = tf.keras.layers.Input(shape=[256,256,3], name='Input_Image')
    #tar = tf.keras.layers.Input(shape=[256,256,3], name='Target_Image')
    
    x = concatenate([inp, tar])
    
    C1 = Conv2D(64, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(x)
    L1 = LeakyReLU()(C1)
    
    C2 = Conv2D(128, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L1)
    B2 = BatchNormalization(axis=1)(C2)
    L2 = LeakyReLU()(B2)
    
    C3 = Conv2D(256, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L2)
    B3 = BatchNormalization(axis=1)(C3)
    L3 = LeakyReLU()(B3)
    
    C4 = Conv2D(512, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L3)
    B4 = BatchNormalization(axis=1)(C4)
    L4 = LeakyReLU()(B4)
    
    last = Conv2D(1, (1,1), kernel_initializer=init, data_format=tf.keras.backend.image_data_format(), activation='sigmoid')(L4) 

    model = Model(inputs=[inp, tar], outputs=last)
    
    return model

discriminator = Discriminator_model()
#discriminator.summary()

def Disc_2():
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        inp = tf.keras.layers.Input(shape=[3, 256, 256], name='input_image')
        tar = tf.keras.layers.Input(shape=[3, 256, 256], name='target_image')
    else:
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    
    #inp = tf.keras.layers.Input(shape=[256,256,3], name='Input_Image')
    #tar = tf.keras.layers.Input(shape=[256,256,3], name='Target_Image')
    
    x = concatenate([inp, tar])
    
    C1 = Conv2D(64, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(x)
    L1 = BatchNormalization(axis=1)(C1)
    
    C2 = Conv2D(128, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L1)
    B2 = BatchNormalization(axis=1)(C2)
    L2 = LeakyReLU()(B2)
    
    C3 = Conv2D(256, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L2)
    B3 = BatchNormalization(axis=1)(C3)
    L3 = LeakyReLU()(B3)
    
    C4 = Conv2D(512, (4,4),padding='same', data_format=tf.keras.backend.image_data_format(), kernel_initializer=init, use_bias=True)(L3)
    B4 = BatchNormalization(axis=1)(C4)
    L4 = LeakyReLU()(B4)
    
    last = Conv2D(1, (1,1), kernel_initializer=init, data_format=tf.keras.backend.image_data_format(), activation='sigmoid')(L4) 

    model = Model(inputs=[inp, tar], outputs=last)
    
    return model

disc_2 = Disc_2()
#disc_2.summary()

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.keras.losses.MAE(target, gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss



def discriminator_loss(disc_real_output, disc_generated_output):
    
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_2_loss(disc_generated_output_2, gen_2_output, target):
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output_2), disc_generated_output_2)
    
    #l1_loss = tf.reduce_mean(tf.keras.losses.MAE(target, gen_2_output))
    
    #ssim = -tf.reduce_mean(tf.image.ssim(target,gen_2_output, max_val=2.0))
    
    l2_loss = tf.reduce_mean(tf.keras.losses.MSE(target, gen_2_output)) 
    
    total_gen_loss = gan_loss + 50*l2_loss

    return total_gen_loss


def disc_2_loss(disc_real_output_2, disc_generated_output_2):
    
    real_loss = loss_object(tf.ones_like(disc_real_output_2), disc_real_output_2)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output_2), disc_generated_output_2)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_2_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_2_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(checkpoint_dir, "TF")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,
                                 generator_2_optimizer=generator_2_optimizer,disc_2_optimizer=disc_2_optimizer,
                                 generator=generator,discriminator=discriminator,generator_2= generator_2,disc_2=disc_2)


@tf.function
def train_step(input_image, target, epoch):
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gen_2_tape, tf.GradientTape() as disc_2_tape:

        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)

        disc_generated_output = discriminator([input_image, gen_output], training=True)



        gen_total_loss = generator_loss(disc_generated_output, gen_output, target)

        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)


        gen_2_output = generator_2(gen_output, training=True) 

        disc_real_output_2 = disc_2([gen_output, target], training=True)

        disc_generated_output_2 = disc_2([gen_output, gen_2_output ], training=True)


        gen_total_loss_2 = generator_2_loss(disc_generated_output_2, gen_2_output, target)

        disc_loss_2 = disc_2_loss(disc_real_output_2, disc_generated_output_2)



    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)

    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

        
    gen_2_gradients = gen_2_tape.gradient(gen_total_loss_2, generator_2.trainable_variables)

    disc_2_gradients = disc_2_tape.gradient(disc_loss_2, disc_2.trainable_variables)

    generator_2_optimizer.apply_gradients(zip(gen_2_gradients, generator_2.trainable_variables))

    disc_2_optimizer.apply_gradients(zip(disc_2_gradients, disc_2.trainable_variables)) 

    return gen_total_loss , disc_loss, disc_loss_2, gen_total_loss_2

def generate_images(test_input):
    
    prediction = generator(test_input)
    
    return  prediction 

def generate_images_2(test_input):
    
    output = generator(test_input)
    
    prediction = generator_2(output)
    
    return  prediction 

def train(input_data, epochs):
    
    for epoch in range(epochs):
        start = time.time()    
        
        for n, (input1, target) in input_data.enumerate():
          
           #if tf.keras.backend.image_data_format() == 'channels_first':
            D1, G1, D2, G2 = train_step(input1, target, epoch)

           
        
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        
            
        print("Epoch: ", epoch)
        print('Disc loss 1:', D1)
        print('Gen loss 1:', G1)
        print('Disc loss 2:', D2)
        print('Gen loss 2:', G2)
        print()
            
        if not os.path.isdir(res_dire_1 + '%04d' % epoch):
                os.makedirs(res_dire_1 + '%04d' % epoch)   
        k = 1
        for inp in test_data.take(50):
            img = generate_images(inp)
            im = np.squeeze(img)
            im1 = im*0.5 + 0.05 # To make pixels betn 0 to 1
            out = np.minimum(np.maximum(im1, 0), 1)
            my = Image.fromarray((im1*255).astype(np.uint8), mode='RGB')
            my.save(res_dire_1 + '%04d/%d.jpg' % (epoch,k))
            k = k + 1
        
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
            
    checkpoint.save(file_prefix = checkpoint_prefix)
        

EPOCHS = 40
E = 20
train(input_data, EPOCHS)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

k = 1
for inp in test_data.take(50):
    img = generate_images_2(inp)
    im = np.squeeze(img)
    im1 = im*0.5 + 0.1 # To make pixels betn 0 to 1
    out = np.minimum(np.maximum(im1, 0), 1)
    my = Image.fromarray((im1*255).astype(np.uint8), mode='RGB')
    my.save(res_dire_2 + '%d.jpg' % k)
    k = k + 1

plt.imshow(im1)

img


# In[ ]:





# In[ ]:




