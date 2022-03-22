import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pickle
import numpy as np
from torch import optim
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from auxiliary.preprocessing import hafemann_preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

class StreamThin(nn.Module):
    def __init__(self):
        super(StreamThin, self).__init__()

        self.stream = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 48, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(48, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.Conv_16 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1)
        self.Conv_32 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.Conv_48 = nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=1)
        self.Conv_64 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.Conv_96 = nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1)

        self.fc_16 = nn.Linear(16, 16)
        self.fc_32 = nn.Linear(32, 32)
        self.fc_48 = nn.Linear(48, 48)
        self.fc_64 = nn.Linear(64, 64)
        self.fc_128 = nn.Linear(128, 128)


        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, reference, inverse):
        for i in range(4):
            reference = self.stream[0 + i * 5](reference)
            reference = self.stream[1 + i * 5](reference)
            inverse = self.stream[0 + i * 5](inverse)
            inverse = self.stream[1 + i * 5](inverse)
            inverse = self.stream[2 + i * 5](inverse)
            inverse = self.stream[3 + i * 5](inverse)
            inverse = self.stream[4 + i * 5](inverse)
            reference = self.attention(inverse, reference)
            reference = self.stream[2 + i * 5](reference)
            reference = self.stream[3 + i * 5](reference)
            reference = self.stream[4 + i * 5](reference)

        return reference, inverse

    def attention(self, inverse, discrimnative):
        """
        注意力模块的设置原理：
        1. 抽出反色流的第二层卷积结果，上采样到和判决流第一层卷积结果相同大小；
        2. 过一层conv提取特征
        3. 与判决流第一层卷积结果做内积再相加，类似skip connection
        4. 分出两路，一路过GAP过FC后再reszie到原大小，成为注意力
           注意力和另一路相乘
        """
        GAP = nn.AdaptiveAvgPool2d((1, 1))
        sigmoid = nn.Sigmoid()
        up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
        conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
        g = conv(up_sample)
        g = sigmoid(g)
        tmp = g * discrimnative + discrimnative
        f = GAP(tmp)
        f = f.view(f.size()[0], 1, f.size()[1])
        fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
        f = fc(f)
        f = sigmoid(f)
        f = f.view(-1, f.size()[2], 1, 1)
        out = tmp * f
        return out


class StreamStandard(nn.Module):
    def __init__(self):
        super(StreamStandard, self).__init__()

        self.stream = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 96, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(96, 128, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.Conv_32 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.Conv_64 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.Conv_96 = nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1)
        self.Conv_48 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1)


        self.fc_32 = nn.Linear(32, 32)
        self.fc_64 = nn.Linear(64, 64)
        self.fc_96 = nn.Linear(96, 96)
        self.fc_128 = nn.Linear(128, 128)


        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, reference, inverse):
        for i in range(4):
            reference = self.stream[0 + i * 5](reference)
            reference = self.stream[1 + i * 5](reference)
            inverse = self.stream[0 + i * 5](inverse)
            inverse = self.stream[1 + i * 5](inverse)
            inverse = self.stream[2 + i * 5](inverse)
            inverse = self.stream[3 + i * 5](inverse)
            inverse = self.stream[4 + i * 5](inverse)
            reference = self.attention(inverse, reference)
            reference = self.stream[2 + i * 5](reference)
            reference = self.stream[3 + i * 5](reference)
            reference = self.stream[4 + i * 5](reference)

        return reference,inverse

    def attention(self, inverse, discrimnative):
        """
        注意力模块的设置原理：
        1. 抽出反色流的第二层卷积结果，上采样到和判决流第一层卷积结果相同大小；
        2. 过一层conv提取特征
        3. 与判决流第一层卷积结果做内积再相加，类似skip connection
        4. 分出两路，一路过GAP过FC后再reszie到原大小，成为注意力
           注意力和另一路相乘
        """
        GAP = nn.AdaptiveAvgPool2d((1, 1))
        sigmoid = nn.Sigmoid()
        up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
        conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
        g = conv(up_sample)
        g = sigmoid(g)
        tmp = g * discrimnative + discrimnative
        f = GAP(tmp)
        f = f.view(f.size()[0], 1, f.size()[1])

        fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
        f = fc(f)
        f = sigmoid(f)
        f = f.view(-1, f.size()[2], 1, 1)
        out = tmp * f
        return out


class Net(nn.Module):
    def __init__(self,mod='thin'):
        super(Net, self).__init__()

        assert mod=='thin' or 'std',"model has only two variant: thin and std"
        if mod =='thin':
            self.stream = StreamThin()
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.stream = StreamThin()
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, inputs):
        half = inputs.size()[1] // 2
        reference = inputs[:, :half, :, :]
        reference_inverse = 255 - reference
        test = inputs[:, half:, :, :]
        del inputs
        test_inverse = 255 - test

        reference, reference_inverse = self.stream(reference, reference_inverse)
        test, test_inverse = self.stream(test, test_inverse)

        cat_1 = torch.cat((test, reference_inverse), dim=1)
        cat_2 = torch.cat((reference, test), dim=1)
        cat_3 = torch.cat((reference, test_inverse), dim=1)

        del reference, reference_inverse, test, test_inverse

        cat_1 = self.sub_forward(cat_1)
        cat_2 = self.sub_forward(cat_2)
        cat_3 = self.sub_forward(cat_3)

        return cat_1, cat_2, cat_3

    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
        out = self.classifier(out)

        return out


class dataset(Dataset):
    def __init__(self, train=True):
        super(dataset, self).__init__()
        if train:
            path= '../../pair_ind/cedar_ind/train_index.pkl'
        else:
            path = '../../pair_ind/cedar_ind/test_index.pkl'

        with open(path, 'rb') as f:
            img_paths = pickle.load(f)
        img_paths=np.array(img_paths)
        self.img_paths = img_paths

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, index):
        ind=self.img_paths[index]
        refer, test, label = ind
        refer_img = cv2.imread(refer, 0)
        test_img = cv2.imread(test, 0)
        refer_img = hafemann_preprocess(refer_img,820,890)
        refer_img=np.expand_dims(refer_img,axis=0)

        test_img = hafemann_preprocess(test_img,820,890)
        test_img=np.expand_dims(test_img,axis=0)
        refer_test = np.concatenate((refer_img, test_img), axis=0)
        return torch.FloatTensor(refer_test), float(label)


def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]

    # majority vote
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.bce_loss = nn.BCELoss()


    def forward(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)
        # print(max(x), max(label))
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)


def eval_stage(model,data_loader):
    """
    用模型预测数据集上的结果，画图并返回预测值
    """
    result=[]
    label=[]
    with torch.no_grad():
        it=iter(data_loader)
        for i in range(len(data_loader)):
            inputs,labels=next(it)
            if cuda:
                inputs,labels=inputs.cuda(),labels.cuda()
            pred=model(inputs)
            result.append((0.3*pred[0]+0.4*pred[1]+0.3*pred[2]).cpu().numpy())
            label.append(labels.cpu())
    result=np.vstack(result)
    label=np.hstack(label)
    draw_fig(result,label)
    return result,label


def draw_fig(pred,label):
    fpr, tpr, thresholds = roc_curve(label,pred, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=pred.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    pred_label=pred.copy()
    pred_label[pred_label>0.5]=1
    pred_label[pred_label<=0.5]=0
    acc_half=(pred_label==label).sum()/label.size

    area = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()

    return area,EER,acc,acc_half

if __name__ == '__main__':
    x = torch.ones(1, 1, 32, 32)
    y = torch.ones(1, 1, 32, 32)
    x_ = torch.ones(1, 1, 32, 32)
    y_ = torch.ones(1, 1, 32, 32)
    # img1=torch.concat([x,x_],axis=1)
    # img2=torch.concat([y,y_],axis=1)
    # input=torch.concat([img1,img2],axis=0)
    model=Net()
    # model = model.cuda()
    # summary(model,(2,155,220),batch_size=32)
    # model(input)



    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 0.0003
    save=False

    np.random.seed(0)
    torch.manual_seed(2)

    cuda = torch.cuda.is_available()

    train_set = dataset(train=True)
    test_set = dataset(train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=0)
    # 检验模型前向传播是否正常
    # test,lab=next(iter(train_loader))
    # model(test)

    if cuda:
        model = model.cuda()
    criterion = loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 创建日志
    writer = SummaryWriter(log_dir='../../logs/scalar')

    if cuda:
        criterion = criterion.cuda()
    iter_n = 0
    t = time.strftime("%m-%d-%H-%M", time.localtime())
    print(len(train_loader))
    for epoch in range(1, EPOCHS + 1):
        print(f'epoch{epoch} start')
        for i, (inputs, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()

            optimizer.zero_grad()

            labels = labels.float()
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            predicted = model(inputs)

            loss = criterion(*predicted, labels)

            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(predicted, labels)

            # writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
            # writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)

            # 每100epochs在测试集上评估结果
            # if i % 100== 0:
            #     with torch.no_grad():
            #         accuracys = []
            #         for i_, (inputs_, labels_) in enumerate(test_loader):
            #             labels_ = labels_.float()
            #             if cuda:
            #                 inputs_, labels_ = inputs_.cuda(), labels_.cuda()
            #             predicted_ = model(inputs_)
            #             accuracys.append(compute_accuracy(predicted_, labels_))
            #         accuracy_ = sum(accuracys) / len(accuracys)
            #         writer.add_scalar(t+'/test_accuracy', accuracy_, iter_n)
            #     print('test loss:{:.6f}'.format(accuracy_))

            iter_n += 1

            if i == 500 and save:
                torch.save(model.state_dict(), '../NetWeights/IDN/IDN.pth')
                break

            print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))
    writer.close()

    result,label=eval_stage(model,test_loader)