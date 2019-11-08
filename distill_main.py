import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils import data as data2
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import os
import copy
import math
import time

from distill_network import *
import PIL.Image as Image
import utils

# define distill loss
class DistillLoss(nn.Module):
    
    def __init__(self):
        super(DistillLoss,self).__init__()
        
    def forward(self, input, target):

        # TODO implement the distill loss here

        y_shape = y.size()[1]
        x_added_dim = x.unsqueeze(1)
        x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
        diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
        totloss = torch.sum(torch.sum(torch.sum(diff)))
        return totloss

# parameters for slide-learn
batch_size=23
num_epochs = 200
learning_rate = 0.001
log_interval = 10
save_interval  = 200
seq_len = 75

distill = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def load_dataloaders():
        
#     # load data
#     batch_size=23
#     [train_ids, train_labels, test_ids, test_labels] = pickle.load(open('slide_6_10.pkl', 'rb'))
#     # print("loading data")
#     # print(len(train_ids), len(train_labels), len(test_ids)) # 1173 1173 253
#     training_dataset = Dataset(train_ids, train_labels)
#     X_train = training_dataset.get_X()
#     Y_train = training_dataset.get_y()
#     # print("example Dataset")
#     # print(training_dataset[0][0].size()) # tuple e.g. (torch.Size([1, 6, 10, 75]), 0)
#     # from iCaRL print(X_train.shape, Y_train.shape, X_test.shape) #(50000, 3, 32, 32) (50000,) (10000, 3, 32, 32)
#     test_dataset = Dataset(test_ids, test_labels)
#     X_test = test_dataset.get_X()
#     Y_test = test_dataset.get_y()
#     # print(X_train.shape)
#     # print(len(X_train))
#     # print(Y_train[0])
#     # print(Y_train.shape)
#     # print(X_train.shape, Y_train.shape, X_test.shape)
#     train_loader = data2.DataLoader(training_dataset, batch_size=batch_size)
#     test_loader = data2.DataLoader(test_dataset, batch_size=batch_size)

#     return train_loader, test_loader


# def slide_learn(iteration):
    
#     dirName = 'slide_info_' + str(iteration)
#     try:
#         # Create target Directory
#         os.mkdir(dirName)
#     except FileExistsError:
#         print('Re-writing the touch')
#     dirName = dirName + '/'
  
#     train_loader, test_loader = load_dataloaders()

#     model = CNN_LSTM(seq_len).to(device)
    
#     # optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     '''
#     def get_total_loss(networks):
#     frcnn_xe_loss = tf.reduce_mean([net.compute_frcnn_crossentropy_loss() for net in networks])
#     tf.summary.scalar('loss/frcnn/class', frcnn_xe_loss)
#     frcnn_bbox_loss = tf.reduce_mean([net.compute_frcnn_bbox_loss() for net in networks])
#     tf.summary.scalar('loss/frcnn/bbox', frcnn_bbox_loss)
#     l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#     tf.summary.scalar('loss/weight_decay', l2_loss)
#     total_loss = frcnn_xe_loss + frcnn_bbox_loss + l2_loss

#     if args.distillation:
#         distillation_xe_loss = tf.reduce_mean([net.compute_distillation_crossentropy_loss() for net in networks])
#         tf.summary.scalar('loss/distillation/class', distillation_xe_loss)
#         distillation_bbox_loss = tf.reduce_mean([net.compute_distillation_bbox_loss() for net in networks])
#         tf.summary.scalar('loss/distillation/bbox', distillation_bbox_loss)
#         total_loss += distillation_xe_loss + distillation_bbox_loss
#     tf.summary.scalar('loss/total', total_loss)
#     return total_loss
#     '''

#     if distill:
#         criterion = nn.CrossEntropyLoss() + DistillLoss()

#     else:
#         criterion = nn.CrossEntropyLoss()

    
#     # set initial hidden state
#     (h_ini, c_ini) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
    
#     epoch_lists = []
#     model.train()
#     for epoch in range(1, num_epochs + 1):
#         for batch_idx, (data, target) in enumerate(train_loader):

#             data = np.expand_dims(data, axis=1)
#             data = torch.FloatTensor(data)
#             #print(target.size())
#             data, target = data.to(device), target.to(device)


#             data, target = Variable(data), Variable(target)
#             #print('Data size:', data.size())

#             optimizer.zero_grad()
            
#             # init hidden states
#             model.init_hidden(h_ini, c_ini)
            
#             output = model(data)
#             #print(target.size(), output.size())

#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#             #if batch_idx % log_interval == 0:
#         epoch_lists.append(loss.item())
#         if epoch % save_interval == 0:
#             torch.save(model.state_dict(), dirName + 'model_epoch_' + str(epoch) +'.ckpt')
         
#     # save epochs
#     pickle.dump(epoch_lists, open(dirName + 'loss.pkl', 'wb'))
    
#     ## check for accuracy
        
#     results = []
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in train_loader:

#         data = np.expand_dims(data, axis=1)
#         data = torch.FloatTensor(data)        
#         data, target = data.to(device), target.to(device)
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += criterion(
#             output, target).item()  # sum up batch loss
#         pred = output.data.max(
#             1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

#     test_loss /= len(train_loader.dataset)
#     results.append( 100.0 * correct.item() / len(train_loader.dataset) )

#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:

#         data = np.expand_dims(data, axis=1)
#         data = torch.FloatTensor(data)        
#         data, target = data.to(device), target.to(device)
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += criterion(
#             output, target).item()  # sum up batch loss
#         pred = output.data.max(
#             1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

#     test_loss /= len(test_loader.dataset)

#     results.append( 100.0 * correct / len(test_loader.dataset) )
    
#     pickle.dump(results, open(dirName + 'results.pkl', 'wb'))

# slide_learn(400)




###########
# Distillation network
###########


def load_datasets():
        
    # load data
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open('slide_6_10_classwise.pkl', 'rb'))
    # print("loading data")
    # print(len(train_ids), len(train_labels), len(test_ids)) # 1173 1173 253
    # interpret the data class by class
    print("train_ids", len(train_ids))
    # print(train_ids)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(train_ids)):
        print("prepare data for class %d" %(i))
        training_dataset = Dataset(train_ids[i], train_labels[i])
        X_train.append(training_dataset.get_X())
        Y_train.append(training_dataset.get_y())
        # print("example Dataset")
        # print(training_dataset[0][0].size()) # tuple e.g. (torch.Size([1, 6, 10, 75]), 0)
        # from iCaRL print(X_train.shape, Y_train.shape, X_test.shape) #(50000, 3, 32, 32) (50000,) (10000, 3, 32, 32)
        test_dataset = Dataset(test_ids[i], test_labels[i])
        X_test.append(test_dataset.get_X())
        Y_test.append(test_dataset.get_y())
        # print(X_train.shape)
        # print(len(X_train))
        # print(Y_train[0])
        # print(Y_train.sh?ape)
    print("load datasets")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) # (23, 50, 6, 10, 75) (23, 50) (23, 12, 6, 10, 75) (23, 12)
    X_val = 0
    Y_val = 0
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# parameters for distill-learning

iteration = 400
iteration_finetune = 300

lr = 0.001
schedules = range(50, iteration, 50)
gamma = 0.5
momentum = 0.9
decay = 0.0001
batchsize=23
num_class = 23
init_num_class = 3 # so that the rest 20 classes can be evenly distributed over periods
cur_num_class = init_num_class
new_num_class = 5
# np.random.seed(100)
# class_order = np.random.permutation(num_class)
class_order = np.arange(num_class)
print("class order")
print(class_order)
memory_K = 300 # (50 per class * 6 classes) #2000 # TODO: tune later
T = 2
dist_ratio = 0.5
stop_acc = 0.998

period_train = (num_class-init_num_class)//new_num_class
memory_size = memory_K//new_num_class # how many samples to load each time
class_old = np.array([], dtype=int)
memory_images = np.zeros(shape=(0, memory_size, 1, 6, 10, 75), dtype = np.uint8)
memory_labels = np.zeros(shape=(0, memory_size), dtype=int)
acc_nvld_basic = np.zeros((period_train))
acc_nvld_finetune = np.zeros((period_train))
crossentropy = nn.CrossEntropyLoss()

# get feature dim
# skip since we are not using pre-trained resnet
input_size = (1, 6, 10, 75)
# newshape=(-1,)+input_size "(-1, 1, 6, 10)" 

# load the dataset
images_train, labels_train, images_val, labels_val, images_test, labels_test = load_datasets()
num_train_per_class = images_train.shape[1]
print("num_train_per_class", num_train_per_class)
# get model state
model_path = 'model/model_slide_e2e_{}.pth'.format((cur_num_class+new_num_class))
flag_model = os.path.exists(model_path)

# gradually increase cur_num_class
for period in range(period_train):
    print('-----------------------------------')
    print('period = %d' % (period))
    if period == 0:
        class_novel = class_order[:(cur_num_class+new_num_class)]
    else:
        class_novel = class_order[cur_num_class:(cur_num_class+new_num_class)]
    print("new classes: ", class_novel)
    # organize the dataset for this period
    # TODO: instead of select sequentially, use representative classes for the init training
    images_new_train = np.reshape(images_train[class_novel], newshape=(-1, 1, 6, 10, 75)) # one more dim on time compared to orginal img
    labels_new_train = np.reshape(labels_train[class_novel], newshape=(-1))
    images_new_test = np.reshape(images_test[class_novel], newshape=(-1, 1, 6, 10, 75))
    labels_new_test = np.reshape(labels_test[class_novel], newshape=(-1))

    num_class_old = class_old.shape[0]

    if period == 0:
        # use the init data to train
        images_combined_train = images_new_train
        labels_combined_train = labels_new_train
        # create the model
        net = Network(num_class=(cur_num_class+new_num_class), num_new_class=0)
        net.to(device)

    else:
        # combine the representatives in memory and newly added data to train
        images_combined_train = np.concatenate((images_new_train, np.reshape(memory_images, newshape=(-1, 1, 6, 10, 75))), axis=0)
        labels_combined_train = np.concatenate((labels_new_train, np.reshape(memory_labels, newshape=(-1))), axis=0)
        # create a new model and make use of previously trained model by state_dict
        net = Network(num_class=num_class_old, num_new_class=new_num_class)
        if period == 1:
            last_net = Network(num_class=num_class_old, num_new_class=0)
        else:
            last_net = Network(num_class=num_class_old, num_new_class=new_num_class)
        last_net.load_state_dict(torch.load(model_path))
        last_net.eval()
        pretrained_dict = last_net.state_dict()
        net_dict = net.state_dict()
        
        # print("last_net dict")
        # print([key for key, value in pretrained_dict.items()])
        # print("net_dict")
        # print([key for key, value in net_dict.items()])
        
        # filter the pretrained dict, drop the last linear weight for net (and subnet if it exists)
        inherit_layers = ['cnn', 'lstm'] # drop the last linear layer
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if any(layer in k for layer in inherit_layers)} 
        # print("filtered pretrained_dict")
        # print([key for key, value in pretrained_dict.items()])

        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        # 2. overwrite entries in the existing state dict
        net_dict.update(pretrained_dict) 
        # 3. load the new state dict
        net.load_state_dict(net_dict)
        net.to(device)
    
    # reorganize the datasets
    images_nvld_test = images_test[np.concatenate((class_old, class_novel), axis=0)]
    images_nvld_test = np.reshape(images_nvld_test, newshape= (-1, 1, 6, 10, 75))
    labels_nvld_test = labels_test[np.concatenate((class_old, class_novel), axis=0)]
    labels_nvld_test = np.reshape(labels_nvld_test, newshape=(-1))

    # skip data augmentation part 

    print("training size = %d"%(labels_combined_train.shape[0]))

    # training
    lrc = lr
    print("current lr = %f" % (lrc))
    acc_training = []
    softmax = nn.Softmax(dim=1).to(device)

    if net.subnet:
        net_old = net.subnet
    else:
        net_old = copy.deepcopy(net)

    # training over iterations/ bactches
    for iter in range(iteration):

        # learning rate
        if iter in schedules:
            lrc *= gamma
            print("current lr = %f" % (lrc))
        
        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=lrc, momentum=momentum, \
            weight_decay=decay, nesterov=True)
        
        # train
        # print("begin train")
        # print(labels_combined_train.shape[0])
        idx_train = np.random.permutation(labels_combined_train.shape[0])
        # print("idx_train")
        # print(idx_train)
        # raise ValueError("stop here")
        loss_avg = 0
        loss_cls_avg = 0
        loss_dist_avg = 0
        acc_avg = 0
        num_exp = 0
        tstart = time.clock()
        batchnum_train = math.ceil(labels_combined_train.shape[0]/batchsize)

        # load model
        if period == 0 and flag_model:
            print("load model %s" % model_path)
            net.load_state_dict(torch.load(model_path))
            # not the case for CNN-LSTM unless u load the model before
            # TODO use the state_dict to load partial model for either first model or subnet
        # print("batchnum_train", batchnum_train)
        # print("idx_train", idx_train)
        for batch in range(batchnum_train):
            if period == 0 and flag_model: # no need training
                num_exp = 1
                break

            if batch == batchnum_train - 1:
                idx = idx_train[batch * batchsize:] # last batch, take all the rest
            else: 
                idx = idx_train[batch * batchsize : (batch+1)*batchsize]
            bs = len(idx) # same as batchsize unless it is the last batch
            img = images_combined_train[idx]
            lab = labels_combined_train[idx]
            # TODO: change onehot encoding to current number of classes only
            # lab_onehot = utils.one_hot(lab, num_class)
            lab_onehot = utils.one_hot(lab, (cur_num_class+new_num_class))

            # transform
            # TODO double check the data format
            # see Network: batch_size, timesteps, C, H, W, sequence_size = x.size() requires 6D
            img = np.expand_dims(img, axis=1)
            img = torch.FloatTensor(img).to(device)
            # print("img input example", img.shape) # torch.Size([23, 1, 1, 6, 10, 75])
            # print(img)
            lab_onehot = torch.from_numpy(lab_onehot).float().to(device)

            # init hidden space for lstm
            (h_ini, c_ini) = (torch.zeros(2, bs, 50).to(device) , torch.zeros(2, bs, 50).to(device))
            net.init_hidden(h_ini, c_ini)

            output = net(img)
            indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
            indices = indices.to(device)
            # if period > 0:
            #     # print the variables for debugging
            #     print("lab")
            #     print(lab)
            #     print("lab_onehot", lab_onehot.size())
            #     print(lab_onehot)
            #     print("output", output.size())
            #     print(output)
            #     
            #     print("indicesi", indices.size())
            #     print(indices)
            prob_cls = softmax(torch.index_select(output, 1, indices))
            lab_onehot = torch.index_select(lab_onehot, 1, indices)
            # if period >0:
            #     print("prob_cls", prob_cls.size(), "lab_onehot", lab_onehot.size())
            #     print("combined old & new indices")
            #     print(indices)
            loss_cls = F.binary_cross_entropy(prob_cls, lab_onehot) # TODO: binary? # classification loss
            # print("batch", batch, "classification loss ", loss_cls)
            # distllation loss for only old class data !
            if period > 0:

                indices = torch.LongTensor(class_old).to(device)
                dist = torch.index_select(output, 1, indices)
                dist = softmax(dist/T)
                # same initialization procedure as net
                (h_ini, c_ini) = (torch.zeros(2, bs, 50).to(device), torch.zeros(2, bs, 50).to(device))
                net_old.init_hidden(h_ini, c_ini)
                output_old = net_old(img)
                ouput_old = torch.index_select(output_old, 1, indices)
                lab_dist = softmax(Variable(output_old, requires_grad=False))

                # # TODO: RH: what is the difference, one is class_old, one is enumerate(lab) if la in class_old
                # if not flag_dist_all: # TODO haven't set the flag before
                #     # only for old class data
                #     indices = [id for id, la in enumerate(lab) if la in class_old]
                #     indices = torch.LongTensor(indices)
                #     indices = indices.cuda()
                #     dist = torch.index_select(dist, 0, indices)
                #     lab_dist = torch.index_select(lab_dist, 0, indices)
                
                # TODO：should only calculate distill loss for those of labels of old classes
                loss_dist = F.binary_cross_entropy(dist, lab_dist)  # distill loss
            else: 
                loss_dist = 0
            # print("distillation loss", loss_dist)
            loss = loss_cls + dist_ratio * loss_dist

            loss_avg += loss.item()
            loss_cls_avg += loss_cls.item()
            if period == 0:
                loss_dist_avg += 0
            else:
                loss_dist_avg += loss_dist.item()

            acc = np.sum(np.equal(np.argmax(prob_cls.cpu().data.numpy(), axis=-1), np.argmax(lab_onehot.cpu().data.numpy(), axis=-1)))
            acc_avg += acc
            num_exp += np.shape(lab)[0]
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
            # because PyTorch accumulates the gradients on subsequent backward passes.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # # add random noise to gradients / weights, not implemented now
            # if gredient_noise_ratio > 0:
            #     for p in net.parameters():
            #         p.data.sub_(gredient_noise_ratio * lrc * torch.from_numpy(
            #             (np.random.random(np.shape(p.data.cpu().data.numpy())) - 0.5)*2).float().cuda())

        loss_avg /= num_exp
        loss_cls_avg /= num_exp
        loss_dist_avg /= num_exp
        acc_avg /= num_exp
        acc_training.append(acc_avg)
        tend = time.clock()
        tcost = tend - tstart

        print('Training Period: %d \t Iter: %d \t time = %.1f \t loss = %.6f \t acc = %.4f' % (period, iter, tcost, loss_avg, acc_avg))

        # test all (novel + old) classes based on logists
        if period > -1:
            idx_test = np.random.permutation(labels_nvld_test.shape[0])
            loss_avg = 0
            acc_avg = 0
            num_exp = 0
            tstart = time.clock()

            batchnum_test = math.ceil(labels_nvld_test.shape[0] / batchsize)
            for bi in range(batchnum_test):
                if bi == batchnum_test - 1:
                    idx = idx_test[bi * batchsize:]
                else:
                    idx = idx_test[bi * batchsize:(bi + 1) * batchsize]
                bs = len(idx)
                img = images_nvld_test[idx]
                lab = labels_nvld_test[idx]
                lab_onehot = utils.one_hot(lab, num_class)

                # normalization
                img = np.expand_dims(img, axis=1)
                img = torch.FloatTensor(img).to(device)

                # init hidden space for lstm
                (h_ini, c_ini) = (torch.zeros(2, bs, 50).to(device) , torch.zeros(2, bs, 50).to(device))
                net.init_hidden(h_ini, c_ini)

                output = net(img)
                indices = torch.LongTensor(np.concatenate((class_old, class_novel), axis=0))
                indices = indices.to(device)
                output = torch.index_select(output, 1, indices)
                output = softmax(output)
                output = output.cpu().data.numpy()
                lab_onehot = lab_onehot[:, np.concatenate((class_old, class_novel), axis=0)]

                acc = np.sum(np.equal(np.argmax(output, axis=-1), np.argmax(lab_onehot, axis=-1)))
                acc_avg += acc
                num_exp += np.shape(lab)[0]

            acc_avg /= num_exp
            tend = time.clock()
            tcost = tend - tstart
            print('Testing novel+old Period: %d \t Iter: %d \t time = %.1f \t\t\t\t\t\t acc = %.4f' % (period, iter, tcost, acc_avg))
            acc_nvld_basic[period] = acc_avg

        # raise ValueError("stop here to check")

        if len(acc_training)>20 and acc_training[-1]>stop_acc and acc_training[-5]>stop_acc:
            print('training loss converged')
            break

    # save model
    if period == 0 and (not flag_model):
        print('save model: %s' % model_path)
        torch.save(net.state_dict(), model_path)
    
    cur_num_class += new_num_class
    # TODO update the memory


    # memory management [random select]
    # reduce old memory
    memory_size = min(num_train_per_class, memory_K // (new_num_class + num_class_old))
    if period >0:
        memory_images = memory_images[:, :memory_size]
        memory_labels = memory_labels[:, :memory_size]
    # add new memory
    memory_new_images = np.zeros((new_num_class, memory_size, 1, 6, 10, 75), dtype=np.uint8)
    memory_new_labels = np.zeros((new_num_class, memory_size), dtype=np.int32)
    # random selection
    for k in range(new_num_class):

        img = images_train[class_novel[k], np.random.permutation(num_train_per_class)[:memory_size]]
        img = np.expand_dims(img, axis=1) # broadcast array from shape (51, 6, 10, 75) to shape (51, 1, 6, 10, 75)
        memory_new_images[k] = img
        memory_new_labels[k] = np.tile(class_novel[k], (memory_size))

    # herding
    # to do

    if period > 0:
        memory_images = np.concatenate((memory_images, memory_new_images), axis=0)
        memory_labels = np.concatenate((memory_labels, memory_new_labels), axis=0)
    else:
        memory_images = memory_new_images
        memory_labels = memory_new_labels


    class_old = np.append(class_old, class_novel, axis=0)

