import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pickle
import numpy as np
from torch import optim
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from auxiliary.preprocessing import hafemann_preprocess

'''
===========================================================
Q：是否要分支共享权重，哪些分支要共享权重？
A；三个分支不共享，pair共享
Q: 人工特征提取是在图片resize前还是resize后进行？
A: resize后吧，resize丧失的图片信息人工特征提取时无法弥补
I: 这份代码的模型结构和Tensorflow版本的基本一致，但仍有以下部分发生了变化：
    1. 在Inception模块中去除了BN层
    2. 图像预处理部分先对原始图像提取轮廓、角点、反色特征，之后在统一
       归一化到相同大小，而在tensorflow版本中是对resize之后的图片进行特征提取
I：可以确认现有模型表现不如之tf版本的Mnet是因为图像预处理部分出了问题，之前的预处理效果更好
   但理论分析来看应该是错误的。
Q: 轮廓、角点特征的提取是应该在图片resize前进行还是resize之后进行呢？
I：如果是在resize前进行的话编程方面存在着困难，因为之后存在着resize带来的插值操作，如果直接拿
   提取的特征图去resize，由于去掉了角点、边缘周围的像素，势必会带来差异。如果是先在原图上处理
   记录坐标，再在resize后的图上圈出倒是可以，但坐标放缩也存在着误差。建议先用最简单的处理方法
   即在resize后的图像上取得特征，后续将这一部分作为扩展实验。
===========================================================
'''


class IncepModule(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(IncepModule, self).__init__()

        self.b1=nn.Sequential(
            nn.Conv2d(in_c,c1,(1,1),(1,1)),
            nn.ReLU(inplace=True),
        )

        self.b2=nn.Sequential(
            nn.Conv2d(in_c,c2[0],(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0],c2[1],(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True)
        )

        self.b3=nn.Sequential(
            nn.Conv2d(in_c,c3[0],(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0],c3[1],kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[1],c3[2],kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(inplace=True)
        )

        self.b4=nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3),stride=(1,1),padding=1),
            nn.Conv2d(in_c,c4,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        b1=self.b1(x)
        b2=self.b2(x)
        b3=self.b3(x)
        b4=self.b4(x)
        coc_b=torch.cat((b1,b2,b3,b4),dim=1)
        return coc_b


class Attention(nn.Module):
    # attention输入size为（39,55),两次下采样
    # 本来里面的conv基本都是采用Resblock，但先试试conv吧，默认inp_c=out_c
    # 前面部分采用Residual attenion的思路，下采样+上采样+skip-connection
    # 后半部分用SCA-CNN的思路，分别获取channel attention和spatial attention
    def __init__(self, inp_c, out_c, size1=(38, 55), size2=(19, 27),K = 64):
        super(Attention, self).__init__()

        self.mpool1=nn.MaxPool2d((2,2),(2,2))  # (19,27)
        self.pre_conv=nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True)
        )
        self.skip_conv=nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True)
        )
        self.mpool2=nn.MaxPool2d((2,2),(2,2))  # (9,18)
        self.inter_block=nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True)
        )
        self.intp1=nn.UpsamplingBilinear2d(size2) # (19,27)
        self.after_conv=nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True)
        )
        self.intp2=nn.UpsamplingBilinear2d(size1) # (38,55)
        self.out_conv=nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True)
        )

        # spatial wise attention parameters
        self.Ws=nn.Parameter(torch.randn(K,out_c))
        self.Whs=nn.Parameter(torch.randn(K,out_c))
        self.bs = nn.Parameter(torch.randn(K,1))
        self.Wi=nn.Parameter(torch.randn(1,K))
        self.bi=nn.Parameter(torch.randn(1))

        # channel wise attention parameters
        self.Wc=nn.Parameter(torch.randn(1,K))
        self.Whc=nn.Parameter((torch.randn(1,K)))
        self.bc=nn.Parameter(torch.randn(K))
        self.Wi_=nn.Parameter(torch.randn(K))
        self.bi_=nn.Parameter(torch.randn(1))

    def forward(self,bench,aux):
        '''
        :type bench:torch.Tensor
        :type aux:torch.Tensor
        '''

        # down sample+up sample
        bench=self.mpool1(bench)
        bench=self.pre_conv(bench)
        skip_bench=self.skip_conv(bench)
        bench=self.mpool2(bench)
        bench=self.inter_block(bench)
        bench=self.intp1(bench)+skip_bench
        bench=self.after_conv(bench)
        bench=self.intp2(bench)
        bench=self.out_conv(bench)

        # spatial wise attention
        V_map_bench=bench.view(bench.shape[0],bench.shape[1],bench.shape[2]*bench.shape[3])
        # V_map_bench=V_map_bench.permute(0,2,1)
        V_map_aux=aux.view(aux.shape[0],aux.shape[1],aux.shape[2]*aux.shape[3])
        # V_map_aux= V_map_aux.permute(0,2,1)
        spa_att=nn.Tanh()((torch.matmul(self.Ws,V_map_bench)+self.bs)+(torch.matmul(self.Whs,V_map_aux)))
        alpha=nn.Softmax(dim=0)(torch.matmul(self.Wi,spa_att)+self.bi)
        spatial_encoded=torch.mul(V_map_aux,alpha)

        # channel wise attention
        V_map_aux=spatial_encoded.view(aux.shape[0],aux.shape[1],aux.shape[2],aux.shape[3])
        V_map_aux=V_map_aux.view(aux.shape[0],aux.shape[1],aux.shape[2]*aux.shape[3],1).mean(dim=2)
        V_map_bench=bench.view(bench.shape[0],bench.shape[1],bench.shape[2]*bench.shape[3],1).mean(dim=2)
        ch_att=nn.Tanh()((torch.matmul(V_map_aux,self.Wc)+self.bc)+(torch.matmul(V_map_bench,self.Whc)))
        beta=nn.Softmax(dim=0)(torch.matmul(ch_att,self.Wi_)+self.bi_)
        beta=beta.view(beta.shape[0],beta.shape[1],1)
        channel_encoded=torch.mul(beta,spatial_encoded)

        channel_encoded=channel_encoded.view(aux.shape[0],aux.shape[1],aux.shape[2],aux.shape[3])

        return channel_encoded

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.contours_feature=nn.Sequential(
            nn.Conv2d(2,64,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(224,32,(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True)
        )
        self.edge_feature=nn.Sequential(
            nn.Conv2d(2,64,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(224,32,(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True)
        )
        self.inverse_feature=nn.Sequential(
            nn.Conv2d(2,64,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,(3,3),(2,2),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(224,32,(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True)
        )
        self.contours_incep=IncepModule(32,64,(48,64),(48,64,64),32)
        self.edge_incep=IncepModule(32,64,(48,64),(48,64,64),32)
        self.inverse_incep=IncepModule(32,64,(48,64),(48,64,64),32)


        # self.att1=Attention(32,32)
        # self.att2=Attention(32,32)

    def forward(self,contour_map,edge_map,inverse_map):
        contour_feature=self.contours_feature[:6](contour_map)
        contour_feature=self.contours_incep(contour_feature)
        edge_feature=self.edge_feature[:6](edge_map)

        edge_feature=self.edge_incep(edge_feature)
        inverse_feature=self.inverse_feature[:6](inverse_map)
        inverse_feature=self.inverse_incep(inverse_feature)

        # lower the dims to reduce the number of parameters
        contour_feature=self.contours_feature[6:9](contour_feature)
        edge_feature=self.edge_feature[6:9](edge_feature)
        inverse_feature=self.inverse_feature[6:9](inverse_feature)

        # contour_feature=self.att1(edge_feature,contour_feature)
        # inverse_feature=self.att2(edge_feature,inverse_feature)

        contour_feature=self.contours_feature[9:](contour_feature)
        edge_feature=self.edge_feature[9:](edge_feature)
        inverse_feature=self.inverse_feature[9:](inverse_feature)

        merged_feature=torch.cat([contour_feature,edge_feature,inverse_feature],dim=1)

        return merged_feature


class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.feature_net=FeatureNet()
        self.judging_net=nn.Sequential(
            nn.Conv2d(6,64,(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,(3,3),(1,1),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            IncepModule(32,64,(48,64),(48,64,64),32),
            nn.AdaptiveAvgPool2d(1)
        )
        self.one_dim=nn.Sequential(
            nn.Linear(224,128),
            nn.ReLU(inplace=True)
        )
        self.logit1=nn.Sequential(
            nn.Linear(128,2),
            nn.ReLU(inplace=True)
        )
        self.logit2=nn.Sequential(
            nn.Linear(128,2),
            nn.ReLU(inplace=True)
        )

    def forward(self,inputs):
        # 使用dataset对数据进行封装时要求数据一定得是tensor类型，不能是list
        # 因此将6个输入都在维度上拼接
        refer_contour=inputs[:,:2,:,:]
        refer_edge=inputs[:,2:4,:,:]
        refer_inverse=inputs[:,4:6,:]

        test_contour=inputs[:,6:8,:,:]
        test_edge=inputs[:,8:10,:,:]
        test_inverse=inputs[:,10:12,:]

        refer_feature=self.feature_net(refer_contour,refer_edge,refer_inverse)
        test_feature=self.feature_net(test_contour,test_edge,test_inverse)

        merged_feature=torch.cat([refer_feature,test_feature],dim=1)

        repr=self.judging_net(merged_feature)
        repr=repr.view(repr.shape[0],repr.shape[1])
        repr=self.one_dim(repr)
        logit1=self.logit1(repr)
        logit2=self.logit2(repr)
        prob=(logit2-logit1)

        return prob

def denoise(img):
    # 改进的预处理方法，不再是直接质心对齐，保持纵横比不变的情况下先将一条边扩大到指定大小，再resize到网络输入
    threshold=cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)[0]#大津法确定阈值
    # img[img>threshold]=255 # 大于阈值的变为白色
    img[img<threshold]=255-img[img<threshold]
    img=255-img # 背景黑色，笔迹白色
    return img

def resize(img, ext_h,ext_w,dst_h=155,dst_w=220):
    h,w,=img.shape
    scale=min(ext_h / h, ext_w / w)
    nh=int(scale*h)
    nw=int(scale*w)
    img=cv2.resize(img,(nw,nh))
    pad_row1=int((ext_h - img.shape[0]) / 2)
    pad_row2= (ext_h - img.shape[0]) - pad_row1
    pad_col1=int((ext_w - img.shape[1]) / 2)
    pad_col2= (ext_w - img.shape[1]) - pad_col1
    img=np.pad(img,((pad_row1,pad_row2),(pad_col1,pad_col2)), 'constant',constant_values=(0,0))
    img=cv2.resize(img,(dst_w,dst_h))
    #img=otsu(img.numpy())
    img=img.astype(np.uint8)
    return img

def hand_crafted(img,*args):
    surf=cv2.xfeatures2d.SURF_create(200)
    kp, des = surf.detectAndCompute(img,None)
    temp_img=np.squeeze(img)
    pt=[i.pt for i in kp]
    pt=np.array(pt)
    loc=np.zeros((pt.shape[0],4))
    loc[:,0]=pt[:,1]-2
    loc[:,1]=pt[:,0]-2
    loc[:,2]=pt[:,1]+2
    loc[:,3]=pt[:,0]+2 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
    loc=loc.astype(int)
    contours_map=np.zeros(temp_img.shape)
    if len(kp)>20:
        for i in range(20):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    else:
        for i in range(len(kp)):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    contours_map=contours_map.astype(np.uint8)
    contours_map=np.expand_dims(contours_map,axis=0) # pytorch里数据格式NCHW，tf是NHWC，记住区别
    if args:
        contours_map=np.concatenate([contours_map,np.expand_dims(args[0],axis=0)],axis=0)
    else:
        contours_map=np.concatenate([contours_map,np.expand_dims(img,axis=0)],axis=0)

    edge_map=cv2.Canny(img,50,150)
    edge_map=np.expand_dims(edge_map,axis=0)
    if args:
        edge_map=np.concatenate([edge_map,np.expand_dims(args[0],axis=0)],axis=0)
    else:
        edge_map=np.concatenate([edge_map,np.expand_dims(img,axis=0)],axis=0)

    inverse_map=cv2.bitwise_not(img)
    inverse_map=np.expand_dims(inverse_map,axis=0)
    if args:
        inverse_map=np.concatenate([inverse_map,np.expand_dims(args[0],axis=0)],axis=0)
    else:
        inverse_map=np.concatenate([inverse_map,np.expand_dims(img,axis=0)],axis=0)

    img_map=np.concatenate([contours_map,edge_map,inverse_map],axis=0)
    return img_map


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
        # if train:
        #     img_paths=img_paths[np.random.permutation(img_paths.shape[0]),:]
        self.img_paths = img_paths

    def __len__(self):
        return self.img_paths.shape[0]

    '''
    def __getitem__(self, index):
        ind=self.img_paths[index]
        refer, test, label = ind
        refer_img = cv2.imread(refer, 0)
        test_img = cv2.imread(test, 0)
        refer_img = denoise(refer_img)
        refer_standard=resize(refer_img,820,890)
        refer_map=hand_crafted(refer_img,refer_standard)
        test_img = denoise(test_img)
        test_standard=resize(test_img,820,890)
        test_map=hand_crafted(test_img,test_standard)
        refer_test = np.concatenate([refer_map, test_map], axis=0)
        return torch.FloatTensor(refer_test), float(label)
    '''

    '''
    def __getitem__(self, index):
        ind=self.img_paths[index]
        refer, test, label = ind
        refer_img = cv2.imread(refer, 0)
        test_img = cv2.imread(test, 0)
        refer_img = denoise(refer_img)
        refer_standard=resize(refer_img,820,890)
        refer_map=hand_crafted(refer_standard,refer_standard)
        test_img = denoise(test_img)
        test_standard=resize(test_img,820,890)
        test_map=hand_crafted(test_standard,test_standard)
        refer_test = np.concatenate([refer_map, test_map], axis=0)
        return torch.FloatTensor(refer_test), float(label)
    '''


    def __getitem__(self, index):
        ind=self.img_paths[index]
        refer, test, label = ind
        refer_img = cv2.imread(refer, 0)
        test_img = cv2.imread(test, 0)
        refer_img=hafemann_preprocess(refer_img,820,890)
        refer_map=hand_crafted(refer_img)
        test_img=hafemann_preprocess(test_img,820,890)
        test_map=hand_crafted(test_img)
        refer_test = np.concatenate([refer_map, test_map], axis=0)
        return torch.FloatTensor(refer_test), float(label)



class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.ce_loss=nn.CrossEntropyLoss()

    def forward(self,pred,label):
        loss=self.ce_loss(pred,label.long())
        return loss

def compute_acc(pred,label):
    res=pred.argmax(dim=1)
    label = label.view(-1,1)
    acc=torch.sum(res==label[:,0]).item()/label.size()[0]
    return acc

def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    area = auc(fpr, tpr)
    print("EER:%f"%EER)
    print('AUC:%f'%area)
    print('ACC(EER_threshold):%f'%acc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()


if __name__=='__main__':

    BATCH_SIZE = 28
    EPOCHS = 1
    LEARNING_RATE = 0.0003

    np.random.seed(0)
    torch.manual_seed(3)

    cuda = torch.cuda.is_available()
    if cuda:
        print("programme is executing on GPU")
    else:
        print("programme is executing on CPU")

    model=Mnet()
    # summary(model,(12,155,220),32)


    train_set=dataset(train=True)
    test_set=dataset(train=False)
    train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    test,lab=next(iter(train_loader))
    pred=model(test)
    loss()(pred,lab)
    test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)


    if cuda:
        model=model.cuda()
    criterion=loss()
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))

    writer=SummaryWriter(log_dir='../../NetWeights/Mnet_v2')

    if cuda:
        criterion=criterion.cuda()
    iter_time=0
    t=time.strftime("%m-%d:%H.%M",time.localtime())
    print(f'The size of training-set:{len(train_loader)}')

    '''
    test=test.cuda()
    lab=lab.cuda()
    for i in range(100):
        torch.cuda.empty_cache()  # 释放GPU显存，不确定有没有用，聊胜于无吧
        optimizer.zero_grad()   # 反向传播时清空

        pred=model(test)

        loss=criterion(pred,lab)
        loss.backward()  # 损失反向传播
        optimizer.step()  # 模型参数更新

        acc=compute_acc(pred,lab)
        writer.add_scalar(t+'/train_loss', loss.item(), iter_time)
        writer.add_scalar(t+'/train_accuracy', acc, iter_time)

        print('iter {}, loss:{:.6f}, accuracy:{:.2f}'.format(i, loss.item(), acc))
    '''




    for epoch in range(1,EPOCHS+1):
        print(f'epoch{epoch} start')
        # for i,(inputs,labels) in enumerate(train_loader):
        it=iter(train_loader)
        for i in range(len(train_loader)):
            inputs,labels=next(it)
            #torch.cuda.empty_cache()  # 释放GPU显存，不确定有没有用，聊胜于无吧

            if cuda:
                inputs,labels=inputs.cuda(),labels.cuda()

            pred=model(inputs)

            loss=criterion(pred,labels)
            optimizer.zero_grad()   # 反向传播时清空
            loss.backward()  # 损失反向传播
            optimizer.step()  # 模型参数更新

            acc=compute_acc(pred,labels)
            writer.add_scalar(t+'/train_loss', loss.item(), iter_time)
            writer.add_scalar(t+'/train_accuracy', acc, iter_time)

            print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{:.4f}%'.format(epoch, EPOCHS, i, loss.item(), acc*100))


            '''
            # 带不动，太慢了，还是放弃吧
            # 每10个batch验证一下模型在测试集上的结果
            if i % 10== 0:
                print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), acc))
                with torch.no_grad():
                    accs = []
                    for i_, (inputs_, labels_) in enumerate(test_loader):
                        labels_ = labels_.float()
                        if cuda:
                            inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                        predicted_ = model(inputs_)
                        accs.append(compute_acc(predicted_, labels_))
                    accuracy_ = sum(accs) / len(accs)
                    writer.add_scalar(t+'/test_accuracy', accuracy_, iter_time)
                print('test acc:{:.6f}'.format(accuracy_))
            '''

            iter_time += 1

            # if i == 500:
            #     torch.save(model.state_dict(), 'Mnet.pth')


    writer.close()





