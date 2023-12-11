import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CGMNIST import CGMNISTDataset
from basic_model import CGClassifier
from utils.utils import setup_seed, weight_init

from ARGS import args



def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def grad_amplitude_diff(v1, v2):
    len_v1 = torch.norm(v1, p=2)
    len_v2 = torch.norm(v2, p=2)
    return len_v1, len_v2, len_v1 - len_v2


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, min_loss, _writer, PATIENCE):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    _a_angle = 0
    _v_angle = 0
    _a_diff = 0
    _v_diff = 0
    _ratio_a = 0

    for step, (spec, image, label) in enumerate(dataloader):

        step_curr = epoch*len(dataloader)+step

        spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
        image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        optimizer.zero_grad()





        # RegBN config. for training mode *************************
        kwargs = {"is_training": True, 
                'n_epoch': epoch, 
                'steps_per_epoch': len(dataloader)}
        a, v, out = model(spec, image, **kwargs)  
        # *************************************************





        # out = softmax(out)
        weight_size = model.fc_out.weight.size(1)
        out_v = (torch.mm(v, torch.transpose(model.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fc_out.bias / 2)
        out_a = (torch.mm(a, torch.transpose(model.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fc_out.bias / 2)

        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)


        if loss.data > min_loss:
            PATIENCE = 0
        else:
            PATIENCE += 1
        loss.backward()

        if step% (len(dataloader)//5) == 0:
            print(f'step: {step} -- loss: {loss.data:0.4f}, loss_gray: {loss_a.data:0.4f}, loss_color: {loss_v.data:0.4f}')


        if args.modulation == 'Normal':
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v
            # pass
        else:
            raise NotImplementedError


        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _ratio_a += ratio_a

        score = sum([softmax(out)[i][label[i]] for i in range(out.size(0))])

    if args.optimizer == 'SGD':
        scheduler.step()
    # f_angle.close()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_angle / len(dataloader), \
           _v_angle / len(dataloader), _ratio_a / len(dataloader), _a_diff / len(dataloader), _v_diff / len(dataloader), PATIENCE


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'CGMNIST':
        n_classes = 10
    else:
        raise NotImplementedError

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)



            # RegBN config. for eval mode *****************************
            kwargs = {"is_training": False}
            a, v, out = model(spec, image, **kwargs)
            # *************************************************




            weight_size = model.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fc_out.bias / 2)


            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():    
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)
    setup_seed(args.random_seed)
    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    else:
        raise NotImplementedError

    model.apply(weight_init)
    model.to(device)


    # load if the pretrained model is avail
    if args.ckpt_path:
        loaded_dict = torch.load(args.ckpt_path)
        state_dict = loaded_dict['model']
        model.load_state_dict(state_dict)
        print('Trained model loaded!')

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-4)

        scheduler = optim.lr_scheduler.StepLR(optimizer, 
            args.lr_decay_step, 
            args.lr_decay_ratio)

    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = None

    if args.dataset == 'CGMNIST':
        train_dataset = CGMNISTDataset(args, mode='train')
        test_dataset = CGMNISTDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support CGMNIST for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:

        trainloss_file = args.logs_path + '/Method-CE-grad-amp' + '/train_loss-' + args.dataset  + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) \
                         + '-epoch' + str(args.epochs) + '-' + args.modulation  \
                         + '-' + str(args.num_frame) + '-optim-' + args.optimizer + '-regbn.txt' if args.regbn else '.txt' 
        
        
        folder = f'Method_CE'
        if args.regbn: 
            folder = folder +'_RegBN' 
        if not os.path.exists(args.logs_path + '/' + folder):
            os.makedirs(args.logs_path + '/' + folder)

        _writer = SummaryWriter(log_dir=args.logs_path + '/' + folder)

        save_path = args.logs_path + '/' + folder

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file) 
        #f_trainloss = open(trainloss_file, 'a')

        best_acc = 0.0
        PATIENCE = 0

        for epoch in range(1,args.epochs+1):
            print('Epoch: {}: '.format(epoch))

            s_time = time.time()
            batch_loss, batch_loss_a, batch_loss_v, a_angle, v_angle, ratio_a, a_diff, v_diff, PATIENCE = train_epoch(args, epoch,
                                                                                                            model,
                                                                                                            device,
                                                                                                            train_dataloader,
                                                                                                            optimizer,
                                                                                                            scheduler,
                                                                                                            args.min_loss,
                                                                                                            _writer,
                                                                                                            PATIENCE
                                                                                                            )

            _writer.add_scalar("LOSS/loss",  batch_loss, epoch)
            _writer.add_scalar("LOSS/loss_gray", batch_loss_a, epoch)
            _writer.add_scalar("LOSS/loss_color", batch_loss_v, epoch)


            e_time = time.time()
            t_time = e_time - s_time
            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            _writer.add_scalar("VAL/acc", acc, epoch)
            _writer.add_scalar("VAL/acc_gray", acc_a, epoch)
            _writer.add_scalar("VAL/acc_color", acc_v, epoch)


            print(f'batch -- loss: {batch_loss:0.4f}, loss_gray: {batch_loss_a:0.4f} loss_color: {batch_loss_v:0.4f}, elpased time: {t_time:.1f}')
            print(f'batch -- acc: {acc:0.3f}, acc_gray: {acc_a:0.3f} acc_color: {acc_v:0.3f}\n')



            if epoch%20 == 0 or (epoch > 10 and acc > best_acc):
                print('Saving model....')
                torch.save(
                    {
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         #'scheduler': scheduler.state_dict()
                    },
                    os.path.join(args.logs_path + '/' + folder, 'epoch-{}-acc-{:0.3f}-gray-{:0.3f}-color-{:0.3f}.pt'.format(epoch, 
                        acc, acc_a, acc_v))
                )
                print(epoch, ': Saved model!!!')
                best_acc = acc

            print('PATIENCE: ', PATIENCE)


            if  PATIENCE > 200:
                print(f"[+] No Progress over {PATIENCE} steps!")
                break

        #f_trainloss.close()
        _writer.close()

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        state_dict = loaded_dict['model']

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))




if __name__ == "__main__":
    main()
