#-*- coding:utf-8 -*- 
import os
import argparse   
import tensorflow as tf
import numpy as np
def parents(op):
  return set(input.op for input in op.inputs)

def children(op):
  return set(op for out in op.outputs for op in out.consumers())
def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""

  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}


def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:  
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,   
            name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  
def getResizeData(img,label):
    # with tf.device('/cpu:0'):
    img=tf.read_file(img)
    img=tf.image.decode_image(img,channels=3)
    #img=tf.image.resize_image_with_crop_or_pad(img,200,400)
    img=tf.to_float(img)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    return img,label

def getData(direct,time):
    # with tf.device('/cpu:0'):
    images=[]
    labels=[]
    for i in os.listdir(direct):
        print(i)
        for j in os.listdir(direct+'/'+i):
            for file in os.listdir(direct+'/'+i+'/'+j+'/'+time):
                images.append(os.path.join(direct+'/'+i+'/'+j+'/'+time,file))
                if(i=='10'):
                    labels.append([1,0,0])
                elif(i=='100'):
                    labels.append([0,1,0])
                else:
                    labels.append([0,0,1])

    images=tf.constant(images)
    labels=tf.constant(labels)
    dataset=tf.data.Dataset.from_tensor_slices((images,labels))
    dataset=dataset.map(getResizeData)
    return dataset
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--frozen_model_filename", default="./frozen_model.pb", type=str, help="Frozen model file to import")  
    args = parser.parse_args()  
    #加载已经将参数固化后的图  
    graph = load_graph(args.frozen_model_filename)  
    direct='./cut'
    time='30'
    trainSet=getData(direct,time)
    trainSet=trainSet.shuffle(buffer_size=100)
    trainSet=trainSet.batch(1)
    trainItrt = trainSet.make_initializable_iterator()
    trainData = trainItrt.get_next()
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    for op in graph.get_operations():  
        print(op.name)  

    # prefix/Placeholder 
    # ...  
    #操作有:prefix/Placeholder  
    #為了預測,我们需要找到需要feed的tensor,需要該tensor的名字  
    #注意prefix/Placeholder僅僅是操作的名字,prefix/Placeholder:0才是tensor的名字  
    x = graph.get_tensor_by_name('prefix/Placeholder:0')  
    y = graph.get_tensor_by_name('prefix/cnn_1/Softmax:0')  
    a = graph.get_operation_by_name('prefix/Placeholder')
    b = graph.get_operation_by_name('prefix/cnn_1/Softmax')
    print(a)
    #print_tf_graph(get_graph())
    ses=tf.Session()
    ses.run(tf.global_variables_initializer())
    ses.run(trainItrt.initializer)
    allcount=0
    correct=0
#    with tf.Session(graph=graph) as sess:
#        while True:
#            try:
#                trainImg,trainLabel=ses.run(trainData)
#                y_out = sess.run(y, feed_dict={x:trainImg    })
#                predic=np.where(y_out[0]==np.max(y_out[0]))
#                label=np.where(trainLabel[0]==1)
#                print('predic',predic[0][0],' label',label[0][0])
#                if(predic==label):
#                    correct+=1
#                allcount+=1
#            except tf.errors.OutOfRangeError:
#                break
#        print(correct/allcount)
#    print ("finish")  