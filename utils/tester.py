import os 
import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,cohen_kappa_score


def test(net, net_head, criterion, data_loader, args, groundTruth=None, visulation=False):

    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    net_head.eval()
    with torch.no_grad():
        for data, _, target in data_loader:
            target = target - 1
            data = data.to(args.device)
            target = target.to(args.device)
            output = net_head(net(data))
            
            test_loss = criterion(output, target).item()
            test_pred = output.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_SD_SSISO(net, net_head, criterion, data_loader, args, groundTruth=None, visulation=False):

    # test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    net_head.eval()
    with torch.no_grad():
        # for data, target in data_loader:
        for data, _, target in data_loader:
            target = target - 1
            data = data.to(args.device)
            target = target.to(args.device)
            output = net_head(net(data))
            
            test_loss = criterion(output, target).item()
            test_pred = output.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            # test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy)) 

    return test_preds, targets


def test_SD_SSISO2(net, criterion, data_loader, args, groundTruth=None, visulation=False):

    # test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        # for data, target in data_loader:
        for data, _, target in data_loader:
            target = target - 1
            data = data.to(args.device)
            target = target.to(args.device)
            output = net(data)
            
            test_loss = criterion(output, target).item()
            test_pred = output.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            # test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy)) 

    return test_preds, targets


def test_SSISO(net, net_head, criterion, data_loader, args, groundTruth=None, visulation=False):

    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    net_head.eval()
    with torch.no_grad():
        for data, _, target in data_loader:
            target = target - 1
            data = data.to(args.device)
            target = target.to(args.device)
            output = net_head(net(data))
            
            test_loss = criterion(output, target).item()
            test_pred = output.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_SMIMO(net, superhead, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for data1, data2, target in data_loader:
            target = target - 1
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            target = target.to(args.device)

            _out1, _out2 = net(data1, data2)
            out1, out2, out3 = superhead(_out1, _out2)
            out = out1 + out2 + out3
            
            test_loss1 = criterion(out1, target).item()
            test_loss2 = criterion(out2, target).item()
            test_loss3 = criterion(out3, target).item()
            test_loss = test_loss1 + test_loss2 + test_loss3

            test_pred = out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


# def test_SMIMO(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
#     test_losses = []
#     test_preds = []
#     targets = []
#     correct = 0

#     net.eval()
#     with torch.no_grad():
#         for data1, data2, target in data_loader:
#             target = target - 1
#             data1 = data1.to(args.device)
#             data2 = data2.to(args.device)
#             target = target.to(args.device)

#             out1, out2, out3 = net(data1, data2)
#             out = out1 + out2 + out3
            
#             test_loss1 = criterion(out1, target).item()
#             test_loss2 = criterion(out2, target).item()
#             test_loss3 = criterion(out3, target).item()
#             test_loss = test_loss1 + test_loss2 + test_loss3

#             test_pred = out.data.max(1, keepdim=True)[1]
#             # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

#             correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
#             test_preds.append(test_pred.cpu())
#             test_losses.append(test_loss)
#             targets.append(target.cpu())

#         test_accuracy = 100. * correct / len(data_loader.dataset)
#         print('Accuracy: {}/{} ({:.2f}%)\n'.format(
#                     correct, len(data_loader.dataset), test_accuracy))
        
#     if visulation and groundTruth.any() != None:
#         hight, width = groundTruth.shape
#         test_preds = torch.cat(test_preds, dim=0).numpy()
#         predict_labels = test_preds.reshape(hight, width)

#         # print(np.unique(predict_labels))
#         draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy.item(), 4)) + "_full"))
#         # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
#         for i in range(hight):
#             for j in range(width):
#                 if groundTruth[i][j] == 0:
#                     predict_labels[i][j] = 0

#         draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy.item(), 4)) + "_label"))    

#     return test_losses, test_preds, targets


def test_SMIMO2(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for data1, data2, target in data_loader:
            target = target - 1
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            target = target.to(args.device)

            out1, out2, out3 = net(data1, data2)
            test_loss = criterion(out1, target) + \
                1 * torch.mean(torch.pow(out2 - data1, 2)) + \
                1 * torch.mean(torch.pow(out3 - data2, 2))
            test_loss = test_loss.item()

            test_pred = out1.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_SMIMO3(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for data1, data2, target in data_loader:
            target = target - 1
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            target = target.to(args.device)

            out1, out2, out3 = net(data1, data2)
            test_loss = criterion(out1, target) + \
                1 * torch.mean(torch.pow(out2 - out1, 2)) + \
                1 * torch.mean(torch.pow(out3 - out2, 2))
            test_loss = test_loss.item()

            test_pred = out1.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_SMISO(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for data1, data2, target in data_loader:
            target = target - 1
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            target = target.to(args.device)

            out = net(data1, data2)
            
            test_loss = criterion(out, target).item()
            test_pred = out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_MMISO(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (data11, data12, data13, data21, data22, data23, target) in enumerate(data_loader):
            target = target - 1
            data11 = data11.to(args.device)
            data21 = data21.to(args.device)
            data12 = data12.to(args.device)
            data22 = data22.to(args.device)
            data13 = data13.to(args.device)
            data23 = data23.to(args.device)
            target = target.to(args.device)

            out = net(data11, data21, data12, data22, data13, data23)
            
            test_loss = criterion(out, target).item()
            test_pred = out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets


def test_MMIMO(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (data11, data12, data13, data21, data22, data23, target) in enumerate(data_loader):
            target = target - 1
            data11 = data11.to(args.device)
            data21 = data21.to(args.device)
            data12 = data12.to(args.device)
            data22 = data22.to(args.device)
            data13 = data13.to(args.device)
            data23 = data23.to(args.device)
            target = target.to(args.device)

            batch_pred, x_cls_cnn, x_cls_trans ,x1_out, \
                x2_out, con_loss, x1c_out, loss_ml, \
                    x_fuse1, x_fuse2, x_transfusion = net(data11, data21, data12, data22, data13, data23)
            if args.pred_flag == 'o_fuse':
                choice_pred = batch_pred
            elif args.pred_flag == 'o_1':
                choice_pred = x1_out
            elif args.pred_flag == 'o_2':
                choice_pred = x2_out
            elif args.pred_flag == 'o_cnn':
                choice_pred = x_cls_cnn    
            elif args.pred_flag == 'o_trans':
                choice_pred = x_cls_trans   
            # print(choice_pred.shape)
            test_loss = criterion(choice_pred, target)  + con_loss #+ criterion(batch_pred_2, batch_target)
            test_pred = choice_pred.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = 100. * correct / len(data_loader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
                    correct, len(data_loader.dataset), test_accuracy))
        
    return test_preds, targets

def get_results(test_preds, targets):
    y_pred_test = [j for i in test_preds for j in i]
    y_targets = [j for i in targets for j in i]
    classification = classification_report(y_targets, y_pred_test, digits=4)
    kappa = cohen_kappa_score(y_targets, y_pred_test)
    # print(classification, kappa)
    return classification, kappa