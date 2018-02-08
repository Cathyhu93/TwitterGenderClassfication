import PyNet as net
from util import *

layer_list = [
                ##Conv1: outputsize = inputsize 48x48
                net.Conv2d(64,5,padding=2,stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Conv2: outputsize = inputsize 48x48
                net.Conv2d(64,5,padding=2,stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Pool1: outputsize = 1/2* inputsize 24x24
                net.MaxPool2d(2,0,2),
                ##Conv3: outputsize = inputsize 24x24
                net.Conv2d(128, 5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Conv4: outputsize = inputsize 24x24
                net.Conv2d(128, 5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Pool2: outputsize = 1/2 * inputsize 12x12
                net.MaxPool2d(2,padding=0,stride=2),
                ##Conv4: outputsize = inputsize 12x12
                net.Conv2d(256, 3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Conv5:out = in 12x12
                net.Conv2d(256, 3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Pool3: output = 6x6
                net.MaxPool2d(2,padding=0,stride=2),
                ##Conv6: out=in 6x6
                net.Conv2d(512, 3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##Conv7: out=in 6x6
                net.Conv2d(512, 3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                ##flatten
                net.Flatten(),
                ##dense
                net.Linear(36,1),
                net.Sigmoid()
             ]

loss_layer = net.Binary_cross_entropy_loss()
optimizer = net.SGD_Optimizer(lr_rate = 0.001,weight_decay=0.0001)
my_model = net.Model(layer_list, loss_layer, optimizer)
my_model.set_input_channel(int(3))
data_set, label_set = dataloader()
max_epoch_num = 20
save_interval = 1
batchsize = 1

for i in range (max_epoch_num):
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set) # design function by yourself
  losses = []
  accs = []
  step = int(6000)  # step is a int number
  for j in range (step):
    # obtain a mini batch for this step training
    # mini batch = 1
    data_bt, label_bt = obtainMiniBatch(batchsize,data_set_cur,label_set_cur,j)  # design function by yourself

    # feedward data and label to the model
    loss, pred = my_model.forward(data_bt, label_bt)

    # save loss and acc
    losses.append(loss)
    acc = getAccuracy(pred,label_bt)
    accs.append(acc)

    # backward loss
    my_model.backward(loss)

    # update parameters in model
    my_model.update_param()

    if j%200 ==0:
        print("Step: "+str(j))
        print("loss: "+str(loss))
        print("acc: "+str(acc))


  '''
    save trained model, the model should be saved as a pickle file
  '''
  if i % save_interval == 0:
    my_model.save_model(str(i) + '.pickle')

  plots(losses,accs,"epoch_"+str(i)+".png")