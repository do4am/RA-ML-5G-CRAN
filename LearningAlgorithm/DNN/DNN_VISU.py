import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflowvisu #modified
import scipy.io
import math


# extract unique labels from dataset
def unique(arr_data):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for i in range(len(arr_data)):
        # check if exists in unique_list or not
        x = arr_data[i]
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


# process raw dataset for testing
def get_data(data_txt):
    # Load Data
    Data = pd.read_csv(data_txt, sep=',', header=None)
    rows, cols = Data.shape
    x = Data.iloc[:, :cols].values  # features
    return x.astype(np.float32)


# 20 % of training set for validating
def get_data_training(data_txt, validating_ratio):
    # Load Data
    Data = pd.read_csv(data_txt, sep=',', header=None)
    rows, cols = Data.shape
    idx = np.arange(0, rows)
    # np.random.seed(0) # the splitting always the same per run
    np.random.shuffle(idx)
    split = int(rows*validating_ratio)
    y_all = Data.iloc[:, cols - 1].values
    labels = unique(y_all)

    # mapping
    y_mapped = []
    for value in y_all:
        y_mapped.append(labels.index(value))

    x_train = Data.iloc[split:, 0:cols - 1].values  # features
    x_val = Data.iloc[0:split, 0:cols - 1].values
    y_train = np.eye(len(labels))[y_mapped[split:]]
    y_val = np.eye(len(labels))[y_mapped[0:split]]

    return x_train.astype(np.float32), y_train.astype(np.int32), \
           x_val.astype(np.float32), y_val.astype(np.int32), \
           labels


def nextbatch(batchsize, datalength):
    idx = np.arange(0, datalength)
    np.random.shuffle(idx)
    return idx[:batchsize]


# sigma_f = softmax ; C = Cross-entrop
testing_txt = 'err40/testRF_S1.csv'
output_txt = 'output/outputDNN_S1_NN.csv'

training_txt = 'err40/trainRF_S5_Multi.csv'
record_train_txt = 'err40/recordTrain-S5-Multi-e40.csv'
record_test_txt = 'err40/recordTest-S5-Multi-e40.csv'
model_txt = 'err40/model-S5-Multi-e40.ckpt'

pretrained_model = 'Trained-model/model.ckpt-S1-Multi'

#classes_mapping = scipy.io.loadmat('data/S1-unique_classes_b4_cluster.mat')['classes_sorted'][0:,0].tolist()
#total_labels = len(classes_mapping)
# same starting point every run
tf.set_random_seed(0)
np.random.seed(0)

# testing_set = scipy.io.loadmat('data/output_S1.mat')['testing_groundtruth_S1'][0:,1:4]
# np.savetxt("data/testRF_S1.csv", testing_set, fmt=['%1.2f','%1.2f','%i'], delimiter=',')

# x_train, y_train, labels = get_data(training_txt, classes_mapping)
# x_test, y_test, _ = get_data(testing_txt, classes_mapping)
x_train, y_train, x_val, y_val, labels = get_data_training(training_txt, validating_ratio=.2)
# x_test, y_test, _ = get_data(testing_txt, labels)

no_labels = len(labels)
# hidden layers and their neurons
# hlayer_neurons = [200, 100]
# hlayer_neurons = [10, 200, 100] # 1
# hlayer_neurons = [10, 300, 200] # 3
# hlayer_neurons = [10, 100, 200]   # 4
# hlayer_neurons = [20, 100, 200]   # 5, record111, lr = #5
# hlayer_neurons = [10, 100, 200]   # 7,8,9,10 - overfiting - pkeep = 0.75
# hlayer_neurons = [10, 100, 200, 300]   # FT-2, FT-3
# hlayer_neurons = [10, 100, 200, 300, 400]   # FT-4
# hlayer_neurons = [10, 100, 200, 300, 500]   # FT-5
# hlayer_neurons = [10, 100, 200, 300, 600, 800]   # FT-6, FT-S1-Multi
# hlayer_neurons = [10, 100, 200, 300, 600, 800]   # FT-6, FT-S1-Multi-2
hlayer_neurons = [10, 100, 200]   # S1-multi from scratch, FT-S1-NN

#hlayer_neurons = [10, 20, 40, 80]   # 6
# hlayer_neurons = [10, 160, 80]

# input X: 2x1 (features), dimension of 1 is neglectable; None indexes the batch_size
X = tf.placeholder(tf.float32, [None, 2])  # input_data
# ground-truth place-holder
Y_true = tf.placeholder(tf.float32, [None, no_labels])
# step of decrease learning rate
step = tf.placeholder(tf.int32)
# learning rate
learning_rate = tf.placeholder(tf.float32)
# dropout at training time only 0.75 but dont fire nodes during test time
pkeep = tf.placeholder(tf.float32)

# initialise weight and bias at each layer
W_1 = tf.Variable(tf.random_normal([x_train.shape[1], hlayer_neurons[0]], stddev=0.1), name='W_1', trainable=True)
b_1 = tf.Variable(tf.zeros([hlayer_neurons[0]]), dtype = tf.float32, name='b_1', trainable=True) # small initialised val = 0.1

W_2 = tf.Variable(tf.random_normal([hlayer_neurons[0], hlayer_neurons[1]], stddev=0.1),name='W_2', trainable=True)
b_2 = tf.Variable(tf.zeros([hlayer_neurons[1]]), dtype = tf.float32, name='b_2', trainable=True) # small initialised val = 0.1

W_3 = tf.Variable(tf.random_normal([hlayer_neurons[1], hlayer_neurons[2]], stddev=0.1), name='W_3', trainable=True)
b_3 = tf.Variable(tf.zeros([hlayer_neurons[2]]), dtype = tf.float32, name='b_3', trainable=True) # small initialised val = 0.1

# W_4 = tf.Variable(tf.random_normal([hlayer_neurons[2], hlayer_neurons[3]], stddev=0.1), name='W_4', trainable=True )
# b_4 = tf.Variable(tf.zeros([hlayer_neurons[3]]), dtype = tf.float32, name='b_4', trainable=True) # small initialised val = 0.1

# W_5 = tf.Variable(tf.random_normal([hlayer_neurons[3], hlayer_neurons[4]], stddev=0.1), name='W_5', trainable=False)
# b_5 = tf.Variable(tf.zeros([hlayer_neurons[4]]), dtype = tf.float32, name='b_5', trainable=False) # small initialised val = 0.1
#
# W_6 = tf.Variable(tf.random_normal([hlayer_neurons[4], hlayer_neurons[5]], stddev=0.1), name='W_6', trainable=True)
# b_6 = tf.Variable(tf.zeros([hlayer_neurons[5]]), dtype = tf.float32, name='b_6', trainable=True) # S1-Multi

W_out = tf.Variable(tf.random_normal([hlayer_neurons[-1], no_labels], stddev=0.1), name='W_out')
b_out = tf.Variable(tf.ones([no_labels])/10, dtype = tf.float32, name='b_out')

# learning model
h1 = tf.nn.relu(tf.add(tf.matmul(X, W_1), b_1))  # hidden layer 1
h1 = tf.nn.dropout(h1, pkeep)

h2 = tf.nn.relu(tf.add(tf.matmul(h1, W_2), b_2))  # hidden layer 2
h2 = tf.nn.dropout(h2, pkeep)

h3 = tf.nn.relu(tf.add(tf.matmul(h2, W_3), b_3))  # hidden layer 3
h3 = tf.nn.dropout(h3, pkeep)

# h4 = tf.nn.relu(tf.add(tf.matmul(h3, W_4), b_4))  # hidden layer 4
# h4 = tf.nn.dropout(h4, pkeep)
#
# h5 = tf.nn.relu(tf.add(tf.matmul(h4, W_5), b_5))  # hidden layer 4
# h5 = tf.nn.dropout(h5, pkeep)
#
# h6 = tf.nn.relu(tf.add(tf.matmul(h5, W_6), b_6))  # hidden layer 4
# h6 = tf.nn.dropout(h3, pkeep)

Ylogits = tf.add(tf.matmul(h3, W_out), b_out) # output layer: if using tf.soft_max_cross_entropy as cost function
Y_predict = tf.nn.softmax(Ylogits)  # output_layer to one-hot vector

# cost function for multi-classification. labels > 2: Cross_entropy
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_true)
loss = tf.reduce_mean(loss)*100

# training accuracy
is_correct = tf.equal(tf.argmax(Y_predict, 1), tf.argmax(Y_true, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training step with decay learning rate
batch_size = 100
iteration = 10000
epoch = 20
iterations = iteration*epoch

# learning_rate = 0.0001 + tf.train.exponential_decay(0.001, step, iteration, 1/math.e) #standard
# learning_rate = 0.0001 + tf.train.exponential_decay(0.05, step, 20000, 1/math.e) # 5,7,8,9,10
# learning_rate = 0.001 + tf.train.exponential_decay(0.01, step, 10000, 1/math.e) # FT-2
# learning_rate = 0.0001 + tf.train.exponential_decay(0.01, step, 10000, 1/math.e) # FT-3, FT-4, FT-5
# learning_rate = 0.0001 + tf.train.exponential_decay(0.01, step, 10000, 1/math.e) # FT-6
# learning_rate = 0.0001 + tf.train.exponential_decay(0.01, step, 1000, 1/math.e) # S1-Multi, S1-Merger,
                                                                                # S5-Multi, S5-Merger, S5-NN
# learning_rate = 0.0001 + tf.train.exponential_decay(0.01, step, 10000, 1 / math.e)  # S1-Multi-2
learning_rate = 0.0001 + tf.train.exponential_decay(0.005, step, 20000, 1/math.e)  # S1-Multi from scratch, S1-Multi from scratch
# learning_rate = 0.0001 + tf.train.exponential_decay(0.001, step, 20000, 1/math.e)  # FT-S1-NN from S1-Multi

#learning_rate = tf.constant(0.0001) # for fine-tuning
# learning_rate = tf.constant(0.0001) # S1-Multi

optimizer = tf.train.AdamOptimizer(learning_rate)  # default lr = 0.001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # default lr = 0.001

# _global_step = tf.train.get_or_create_global_step()
train_step = optimizer.minimize(loss)

# set these after setting tf.variable
init_g = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()
# init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_g)

# pre-trained last layer
# saver_restore = tf.train.Saver()
saver_finetuning = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3,
                                   'b_1': b_1, 'b_2': b_2, 'b_3': b_3}) # 3layers

# saver_finetuning = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_4': W_4,
#                                    'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_4': b_4}) # 4 layers

# saver_finetuning = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_4': W_4, 'W_5': W_5,
#                                    'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_4': b_4, 'b_5': b_5}) # 5 layers

saver_current = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_out' : W_out,
                                'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_out' : b_out}
                               , max_to_keep=100)

# saver_current = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_4': W_4, 'W_out' : W_out,
#                               'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_4': b_4, 'b_out' : b_out}
#                              , max_to_keep=100)
#
# saver_current = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_4': W_4, 'W_5': W_5, 'W_out' : W_out,
#                               'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_4': b_4, 'b_5': b_5, 'b_out' : b_out}
#                              , max_to_keep=100)

# saver_current = tf.train.Saver({'W_1': W_1, 'W_2': W_2, 'W_3': W_3, 'W_4': W_4, 'W_5': W_5, 'W_6': W_6, 'W_out' : W_out,
#                                 'b_1': b_1, 'b_2': b_2, 'b_3': b_3, 'b_4': b_4, 'b_5': b_5, 'b_6': b_6, 'b_out' : b_out}
#                                , max_to_keep=100)

def testing_step(testing_dir, output_dir):
    labels_remapped = []
    x_test = get_data(testing_dir)
    labels_mapped = sess.run(tf.argmax(Y_predict, 1), feed_dict={X: x_test, pkeep: 1.0})
    for lb in labels_mapped:
        labels_remapped.append(labels[lb])
    np.savetxt(output_dir, labels_remapped, fmt=['%i'], delimiter=',')
    print('Predicted labels saved to : {}'.format(output_dir))


# matplotlib visualisation
# datavis = tensorflowvisu.MnistDataVis()

record_train = []
record_test = []


def training_step(i, update_test_data, update_train_data):
        idx = nextbatch(batch_size, x_train.shape[0])
        batch_X = x_train[idx, :]
        batch_Y = y_train[idx, :]

        if update_train_data:
            a, l, lr = sess.run([accuracy, loss, learning_rate], feed_dict = {X: batch_X, Y_true: batch_Y, pkeep: 1.0, step: i})
            print('train accuracy:' + str(a) + ' train loss: ' + str(l) + ' (lr: ' + str(lr) + ')')
            # datavis.append_training_curves_data(i, a, l)
            record_train.append([i, a, l, lr])

        if update_test_data:
            a, l = sess.run([accuracy, loss], feed_dict = {X: x_val, Y_true: y_val, pkeep: 1.0})
            print('epoch : ' + str(i//iteration) + ' i : ' + str(i) + ' validate accuracy:' + str(a) + ' validate loss: ' + str(l))
            print('\n')
            # datavis.append_test_curves_data(i, a, l)
            record_test.append([i, a, l])

        # back propagation training_step
        sess.run(train_step, feed_dict={X: batch_X, Y_true: batch_Y,  pkeep: 1.0, step: i})
        if i % 50000 == 0:
            save_path = saver_current.save(sess, model_txt, global_step=i)
            print('save model to: {}'.format(save_path))
            np.savetxt(record_train_txt, record_train, fmt=['%i', '%1.3f', '%1.3f', '%1.5f'], delimiter=',')
            np.savetxt(record_test_txt, record_test, fmt=['%i', '%1.3f', '%1.3f'], delimiter=',')



# restore pre-trained model
# saver_finetuning.restore(sess, pretrained_model) # to fine-tune pre-trained model
# saver_current.restore(sess, pretrained_model) # to use pre-trained model
print('Ready to train and/or test\n')
print('Labels:' + str(len(labels)))

# Training and validating with visualizing effect
# datavis.animate(training_step, iterations=iteration*epoch + 1,
#                 train_data_update_freq=int(batch_size/5),
#                 test_data_update_freq=batch_size,
#                 more_tests_at_start=True)

# Training and validating without visualizing
for i in range(0, iterations + 1):
    training_step(i, i%100 == 0, i%20 == 0)

#
# Testing
# testing_step(testing_txt, output_txt)

sess.close()
print('Done !\n')
