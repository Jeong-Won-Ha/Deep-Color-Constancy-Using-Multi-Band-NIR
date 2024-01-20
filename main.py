import os
import torch.nn as nn
import torch.optim as optim
import numpy as np

import easydict
import socket
import time
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataloader import DatasetFromFolder
from model_2branch import Full_model as Full_model
# from model import FC4 as Full_model
from loss import *
from torchvision.transforms import *

opt = easydict.EasyDict({
    "batchSize": 16,  # batch size
    "lr": 1e-4,  # learning rate
    "patch_size": 256,

    "start_iter": 1,
    "nEpochs": 2000,  # training final epoch
    "snapshots": 20,  # weight save period

    "data_dir": "C:/NIR_dataset_0205/",  # dataset directory

    "model_type": "_FC4",  # model name

    "save_folder": "./weights/bright_rgb_proposed_3ch/",  # weight save directory
    "resume": False,
    "pretrained": False,
    "gpu_mode": True,
    "threads": 1,
    "seed": 123,#20211222
    "gpus": 1,

    # "input_dir": "H:\dataset_211007",  # test dataset dir
    # "test_dataset": "Test.pt",
    "testBatchSize": 1,

    "output": "./results/bright_rgb_proposed_3ch/",  # save folder
    # "testresult": "same_frame",
})

checkpoint_name = './weights/RGB_AlexNet/'
checkpoint_name2 = os.path.join(checkpoint_name + 'DESKTOP-NJ37VPANIR_RGB_epoch_199.pth')

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()

gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
# print(opt)

label = np.zeros(shape=(257, 3), dtype=np.float)
label2 = np.zeros(shape=(78, 3), dtype=np.float)

###################### Ground truth illuminant #################################
label[0:77,:] = [0.748821368466364, 0.567288158220369, 0.342710816390589]
label[77:156,:] = [0.794080769118038, 0.538175815166851, 0.282493405385683]
label[156:233,:] = [0.834806408654636, 0.505058216168657, 0.219122017035446]
label[233:241,:] = [0.833468706405267, 0.506333341309169, 0.221261074122458]
label[241:249,:] = [0.79648869429934, 0.536228824980606, 0.279400084311461]
label[249:257,:] = [0.755398531287944, 0.563615487844529, 0.334231418017203]

label2[0:23,:] = [0.748821368466364, 0.567288158220369, 0.342710816390589]
label2[23:46,:] = [0.794080769118038, 0.538175815166851, 0.282493405385683]
label2[46:69,:] = [0.834806408654636, 0.505058216168657, 0.219122017035446]
label2[69:72,:] = [0.833468706405267, 0.506333341309169, 0.221261074122458]
label2[72:75,:] = [0.79648869429934, 0.536228824980606, 0.279400084311461]
label2[75:78,:] = [0.755398531287944, 0.563615487844529, 0.334231418017203]
###############################################################################################


def train(epoch):
    epoch_loss = 0
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        # input1, input2, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        input1, nir_input, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

        if cuda:
            input1 = input1.cuda(gpus_list[0])
            nir_input = nir_input.cuda(gpus_list[0])
            # map_input = map_input.cuda(gpus_list[0])
            gt = gt.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()

        input = torch.cat((input1, nir_input),dim=1)
        # input = input1

        prediction, _, _ = model(input)

        loss = criterion(prediction, gt)
        # t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        t1 = time.time()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.data,
                                                                                 (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def eval(epoch):
    avg_angle = 0
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input1, nir, gt, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
        if cuda:
            input1 = input1.cuda(gpus_list[0])
            gt = gt.cuda(gpus_list[0])
            nir = nir.cuda()

        t0 = time.time()

        with torch.no_grad():
            input = torch.cat((input1,nir),dim=1)
            prediction,_,_ = model(input)

        angle_error = criterion(prediction, gt)
        avg_angle += angle_error.data

        t1 = time.time()
        prediction = prediction / torch.norm(prediction)

    average = avg_angle / len(testing_data_loader)

    log('Epoch[%d] : Test Avg loss = %.4f \n' % (epoch, average), logfile)


def checkpoint(epoch):
    model_out_path = opt.save_folder + hostname + opt.model_type + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def transform():
    return Compose([
        ToTensor(),
    ])

if __name__ == '__main__':

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = DatasetFromFolder(opt.data_dir, gt_ilu=label,patch_size=opt.patch_size,folder='train/',Num_ch='nir_3ch/',
                                  isTrain=True, transform=transform())
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    print('===> Loading datasets')
    test_set = DatasetFromFolder(opt.data_dir,gt_ilu=label2, patch_size=opt.patch_size,transform=transform(),
                                 folder='test/',Num_ch='nir_3ch/',isTrain=False)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

    print('===> Building model ', opt.model_type)

    # model = FC4(in_ch=3+1)
    model = Full_model(in_ch=3+3)

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = angular_loss

    if cuda:
        model = model.cuda(gpus_list[0])
        # criterion = criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    logfile = opt.save_folder + 'eval.txt'

    if opt.resume:  # resume from check point, load once-trained data
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        checkpoint_load = torch.load(checkpoint_name2)
        model.load_state_dict(checkpoint_load)


    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        if (epoch + 1) % (opt.snapshots) == 0:
            checkpoint(epoch)
            eval(epoch)
        if (epoch + 1) == 300:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        if (epoch + 1) == 700:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % 40 == 0:
            eval(epoch)