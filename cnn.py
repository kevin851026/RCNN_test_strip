import tensorflow as tf
import numpy as np
import os

def getResizeData(img,label):
    # with tf.device('/cpu:0'):
    img=tf.read_file(img)
    img=tf.image.decode_image(img,channels=3)
    img=tf.image.resize_image_with_crop_or_pad(img,100,500)
    # img=tf.image.resize_images(img, size=(672,960),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img=tf.to_float(img)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    return img,label

def getData(direct):
    # with tf.device('/cpu:0'):
    drct=''
    images=[]
    labels=[]
    for i in [10,100,1000]:
        drct=direct+str(i)
        for file in os.listdir(drct):
            images.append(os.path.join(drct,file))
            if i==10:
                labels.append([1,0,0])
            elif i==100:
                labels.append([0,1,0])
            else:
                labels.append([0,0,1])
    images=tf.constant(images)
    labels=tf.constant(labels)

    # img=tf.read_file(images[0])
    # img=tf.image.decode_jpeg(img,channels=3)
    # img=tf.image.resize_images(img, size=(200, 400),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # print(img)
    # ses=tf.Session()
    # ses.run(tf.global_variables_initializer())
    # img=ses.run(img)
    # plt.imshow(img)
    # plt.show()

    dataset=tf.data.Dataset.from_tensor_slices((images,labels))
    dataset=dataset.map(getResizeData)
    return dataset

def convNet(x_in,dropout,reuse,isTraining):
    with tf.variable_scope('cnn',reuse=reuse):
        inputLayer=tf.reshape(x_in,[-1,100,500,3])
        conv1=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[10,10],strides=2,padding='same',activation=tf.nn.relu,reuse=None,name='conv1')
        # conv1_re=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu,reuse=True,name='conv1')
        pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])

        conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu)
        pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])

        # conv3=tf.layers.conv2d(inputs=pool2,filters=200,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu)
        # pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])
        # print(pool2)
        flat=tf.reshape(pool2,[-1,12*62*64])

        dense1=tf.layers.dense(inputs=flat, units=800, activation=tf.nn.relu)
        dropout1=tf.layers.dropout(inputs=dense1, rate=dropout, training=isTraining)
        dense2=tf.layers.dense(inputs=dropout1, units=800, activation=tf.nn.relu)
        dropout2=tf.layers.dropout(inputs=dense2, rate=dropout, training=isTraining)
        # dense3=tf.layers.dense(inputs=dropout2, units=400, activation=tf.nn.relu)
        # dropout3=tf.layers.dropout(inputs=dense3, rate=dropout, training=isTraining)
        logits=tf.layers.dense(inputs=dropout2, units=3)
        #for test
        logits=tf.nn.softmax(logits) if not isTraining else logits

    return logits


if __name__=='__main__':
    direct='./resize4/time30_4/'
    trainSet=getData(direct)
    trainSet=trainSet.shuffle(buffer_size=1200)
    trainSet=trainSet.batch(64)
    trainItrt = trainSet.make_initializable_iterator()
    trainData = trainItrt.get_next()

    # with tf.device('/cpu:0'):
    # direct2='./cardVali'
    # testSet=getData(direct2)
    # testSet=testSet.batch(1993)
    # testItrt = testSet.make_initializable_iterator()
    # testData=testItrt.get_next()


    x_in=tf.placeholder(tf.float32, [None,100,500,3])
    y_in=tf.placeholder(tf.float32, [None,3])
    #for train
    trainGraph=convNet(x_in,0.6,False,True)
    #for test 
    # with tf.device('/cpu:0'):
    testGraph=convNet(x_in,1.,True,False)

    # oneHotLabels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)##################
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=trainGraph)
    optimizer=tf.train.AdamOptimizer(0.000001).minimize(loss)

    # lossT = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=testGraph)
    # optimizerT=tf.train.AdamOptimizer(0.001).minimize(lossT)

    correct_prediction = tf.equal(tf.argmax(testGraph, 1), tf.argmax(y_in, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()#config=tf.ConfigProto(log_device_placement=True)
    sess.run(tf.global_variables_initializer())
    for op in tf.get_default_graph().get_operations():
        print( op.name) 
   for i in range(300):
       print('--------------------------------------')
       print(i)
       sess.run(trainItrt.initializer)
       while True:
           try:
               trainImg,trainLabel=sess.run(trainData)
               sess.run(optimizer,feed_dict={x_in:trainImg,y_in:trainLabel})
               ls=sess.run(loss,feed_dict={x_in:trainImg,y_in:trainLabel})
               print('loss = '+'{:.6f}'.format(ls))
               train_acc=sess.run(accuracy,feed_dict={x_in:trainImg,y_in:trainLabel})
               print('train acc:'+'{:.4f}'.format(train_acc))
           except tf.errors.OutOfRangeError:
               break
       print('******************************************')#
       # with tf.device('/cpu:0'):
       # sess.run(testItrt.initializer)
       # testImg,testLabel=sess.run(testData)
       # ls=sess.run(loss,feed_dict={x_in:testImg,y_in:testLabel})
       # print('loss = '+'{:.6f}'.format(ls))
       # test_acc=sess.run(accuracy,feed_dict={x_in:testImg,y_in:testLabel})
       # print('test acc:'+'{:.4f}'.format(test_acc))#
       if i%10==0:
           saver = tf.train.Saver()
           savePath='./model/'+str(i)
           mdlName='model'+str(i)
           saver.save(sess, os.path.join(savePath,mdlName))
