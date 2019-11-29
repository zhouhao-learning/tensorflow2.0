import collections
import math
import random
import json
import numpy as np
from six.moves import xrange
import tensorflow as tf

with open("dictionary.json")as f1:
    dictionary = json.load(f1)
with open("reverse_dictionary.json")as f2:
    reverse_dictionary = json.load(f2)


valid_word = ['萧炎','灵魂','火焰','萧薰儿','药老','天阶',"云岚宗","乌坦城","惊诧"]

valid_size = 9
valid_examples =[dictionary[li] for li in valid_word]
valid_dataset = np.array(valid_examples,dtype="int32")  # tf2.0中不再需要tensor或constant,直接转成array
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./model/-20.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))  # 只需要指定目录就可以恢复所有变量信息
    gragh = tf.compat.v1.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
    # # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    #  获取需要进行计算的operator
    # 虽然在定义模型时定义了输出层名为"embedding",但是通过打印节点名称才发现，最后一层输出名是"embedding/Read/ReadVariableOp:0'"
    embeddings = sess.graph.get_tensor_by_name('embedding/Read/ReadVariableOp:0')
    feed_dict = {input_x: valid_dataset}
    embeddings = sess.run(embeddings, feed_dict)
    norm = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(embeddings), axis=1, keepdims=True), name="norm")
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(params=normalized_embeddings, ids=valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    sim = similarity.eval()
    for i in xrange(valid_size):
        valid_word = reverse_dictionary[str(valid_examples[i])]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[:top_k]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
            close_word = reverse_dictionary[str(nearest[k])]
            log_str = "%s %s," % (log_str, close_word)
        print(log_str)
