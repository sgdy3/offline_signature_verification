"""
基于VIT方法的脱机签名认证方法
"""
import os.path
import argparse
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import  Dataset,DataLoader
import numpy as np
import cv2
from auxiliary.preprocessing import  hafemann_preprocess
from torch import optim
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,head_num=8,k_dim=64,v_dim=64):
        super(MultiHeadAttention, self).__init__()

        self.k_dim=k_dim
        self.v_dim=v_dim
        self.head_num=head_num
        assert input_dim==head_num*v_dim,"output size of attention block shoul be equal to the input size"
        self.K_mapping = nn.Sequential(
            nn.Linear(input_dim,k_dim*head_num),
            Rearrange('b n (h d) -> b h n d',h=head_num)
        )
        self.Q_mapping = nn.Sequential(
            nn.Linear(input_dim,k_dim*head_num),  # Q,K的大小需要一致，因为要相乘
            Rearrange('b n (h d) -> b h n d',h=head_num)
        )
        self.V_mapping = nn.Sequential(
            nn.Linear(input_dim,v_dim*head_num),  # V的大小可以随意，但v*head_num=inpur，因为有残差连接
            Rearrange('b n (h d) -> b h n d',h=head_num)
        )
        self.activation=nn.Softmax(dim=-1)


    def forward(self,embedding):
        Q = self.Q_mapping(embedding)
        K = self.K_mapping(embedding)
        V = self.V_mapping(embedding)

        out = self.activation(torch.matmul(Q,K.transpose(-1,-2))/self.k_dim**(0.5))
        out = torch.matmul(out,V)
        out = rearrange(out,'b h n d -> b n (h d)')

        return out


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,drop_out=0):
        super(MLP, self).__init__()

        self.model=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim,input_dim),  # residual connection
            nn.Dropout(drop_out)
        )
    def forward(self,img):
            out = self.model(img)
            return out


class Vit(nn.Module):
    def __init__(self, depth, img_shape, embedding_dim, p_w, p_h,num_class,mlp_dim=128,head_num=8,k_dim=64,v_dim=64):
        super(Vit, self).__init__()

        assert (img_shape[0]%p_h==0)&(img_shape[1]%p_w==0),"patch size is not proportional to the img size"
        self.patch_num= (img_shape[0]//p_h)*(img_shape[1]//p_w) # patch的数目
        self.patch_embedding=nn.Sequential(
            Rearrange('b c (h p1) (w p2)-> b (h w) (p1 p2 c)',p1=p_h,p2=p_w), # 将输入图像分成多个flatten的patches
            nn.Linear(p_h*p_w, embedding_dim),
        )

        self.encoder=nn.ModuleList([])
        for i in range(depth):
            self.encoder.append(
                nn.ModuleList([
                    nn.LayerNorm(embedding_dim),
                    MultiHeadAttention(embedding_dim,head_num, k_dim, v_dim),
                    nn.LayerNorm(embedding_dim),
                    MLP(embedding_dim,mlp_dim),
                ])
            )

        self.cls_token=nn.Parameter(torch.randn(1,1,embedding_dim)) # embedding前附加值判决类别
        self.pose_embedding=nn.Parameter(torch.randn(1,self.patch_num+1,embedding_dim))

        self.MLPClassHead = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim,num_class),
        )

        self.MLPAuthHead = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

        self.feature_out=[]

    def forward(self,img):
        embedding = self.patch_embedding(img)  # 将输入图片按patch flatten，格式由b*c*h*w变成b* (p*p) * (h%p*w%p*c)
        batch_size,patch_num,_ = embedding.shape
        cls_token = repeat(self.cls_token,'() n d -> b n d',b=batch_size)  # flatten的bpatch前加入token最后作为输出的类别信息
        embedding = torch.cat([cls_token,embedding],dim=1)
        embedding = embedding+self.pose_embedding  # 添加位置信息

        for ln1,attention,ln2,mlp in self.encoder:
            embedding = ln1(embedding)
            embedding = attention(embedding)+embedding
            embedding = ln2(embedding)
            embedding = mlp(embedding)+embedding

        embedding = embedding[:,0]  # 取出classs token进行后续判别
        #  测试阶段记录features
        if not self.training:
            self.feature_out.append(embedding.detach())
        class_labs = self.MLPClassHead(embedding)
        auth_labs = self.MLPAuthHead(embedding)
        return class_labs, auth_labs

class dataset(Dataset):
    def __init__(self,path,labs):
        super(dataset, self).__init__()
        self.img_path = path
        self.labs = labs

    def __len__(self):
        return self.img_path.shape[0]

    def __getitem__(self, item):
        ind = self.img_path[item] # 某张图片的的path
        user_lab,auth_lab = self.labs[item]
        img = cv2.imread(ind,0)
        img = hafemann_preprocess(img,820,890)
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img),user_lab.astype(np.float32),auth_lab.astype(np.float32)


class model_loss(nn.Module):
    def __init__(self,alpha=0.99):
        super(model_loss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.verify_loss = nn.BCELoss()
        self.alpha=alpha

    def forward(self,pred_class,pred_auth,class_lab,auth_lab):
        return self.alpha*self.class_loss(pred_class,class_lab.long())+(1-self.alpha)*self.verify_loss(pred_auth.flatten(),auth_lab)


def compute_accuracy(class_scores,true_lab,auth_scores,auth_lab):
    pred_lab = torch.argmax(class_scores,1)
    class_correct = (pred_lab==true_lab).sum()
    pred_auth = torch.where(auth_scores>0.5,1,0)
    auth_correct = (pred_auth.flatten()==auth_lab).sum()
    return class_correct/true_lab.shape[0],auth_correct/auth_lab.shape[0]



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',choices=['train','test'],default='train',help='training or testing phase')
    args = parser.parse_args()

    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    save_path =  '../../weights/sig_transformer'
    print(os.path.abspath(save_path))
    np.random.seed(3)
    train_user=np.random.choice(range(1,56),50,replace=False)
    test_user=np.arange(1,56)[~np.isin(np.arange(1,56),train_user)] # 得到测试集用户

    # 训练集划分
    train_path=[]
    train_label=[]
    label_dict=dict(zip(train_user,range(train_user.shape[0])))  # 原用户标签映射到用户数量range之内
    for user in train_user:
        for id in range(1,25):
            train_path.append(org_path%(user,id))
            train_path.append(forg_path%(user,id))
            train_label.append([label_dict[user],1])
            train_label.append([label_dict[user],0])
    train_path=np.array(train_path)
    train_label=np.array(train_label)


    # 测试集划分
    test_path=[]
    test_label=[]
    for user in test_user:
        for id in range(1,25):
            test_path.append(org_path%(user,id))
            test_path.append(forg_path%(user,id))
            test_label.append([user,1])
            test_label.append([user,0])  # 不计算损失，无需进行标签映射
    test_path=np.array(test_path)
    test_label=np.array(test_label)

    BATCHSIZE=32
    EPOCHS=10
    LEARNING_RATE=0.0003
    cuda=torch.cuda.is_available()
    v = Vit(depth=8,img_shape=(150,220),embedding_dim=512,p_w=20,p_h=30,num_class=50)

    train_dataset=dataset(train_path,train_label)
    train_loader=DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True,num_workers=0)
    test_dataset=dataset(test_path,test_label)
    test_loader=DataLoader(test_dataset,batch_size=BATCHSIZE,shuffle=False,num_workers=0)

    if cuda:
        v = v.cuda()

    criterion = model_loss(alpha=0.9)
    if cuda:
        criterion = criterion.cuda()
    optimizer = optim.Adam(v.parameters(),lr=LEARNING_RATE)

    if args.mode == 'train':
        v.train()
        print("training stage start")
        # summary(v,(1,150,220),32)
        for epoch in range(1, EPOCHS + 1):
            print(f'epoch{epoch} starts')

            it = iter(train_loader)

            for i in range(len(train_loader)):
                img,class_lab,auth_lab = next(it)
                if cuda:
                    img = img.cuda()
                    class_lab = class_lab.cuda()
                    auth_lab = auth_lab.cuda()

                pred_class,pred_auth = v(img)
                loss =criterion(pred_class,pred_auth,class_lab,auth_lab)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc =compute_accuracy(pred_class,class_lab,pred_auth,auth_lab)

                print('Epoch[{}/{}], iter {}, loss:{:.6f},class_acc:{:.4f},auth_acc:{:.4f}'.format(epoch, EPOCHS, i, loss.item(),acc[0],acc[1]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(v.state_dict(),save_path+'/sig_transformer.pt')
    else:
        v.eval()
        v.load_state_dict(torch.load(save_path+'/sig_transformer.pt'))
        print("testing stage start")


        # 比较有意思的中间层输出提取方法，解析见后续博客
        # features=[]
        # def hook(module,input,output):
        #     features.append(input)
        #     return None
        # v.MLPClassHead.register_forward_hook(hook)

        #  提取训练集用户的特征作为反例
        it = iter(train_loader)
        for i in range(len(train_loader)):
            test_img,class_lab,auth_lab = next(it)
            if cuda:
                test_img = test_img.cuda()
            v(test_img)
        neg_vecs=torch.concat(v.feature_out)
        neg_vecs=neg_vecs.cpu().numpy()
        v.feature_out=[]

        #  提取测试集用户的特征
        it = iter(test_loader)
        for i in range(len(test_loader)):
            test_img,class_lab,auth_lab = next(it)
            if cuda:
                test_img = test_img.cuda()
            v(test_img)
        test_vec=torch.concat(v.feature_out)
        test_vec = test_vec.cpu().numpy()

        result=[]
        #  为每个用户设计用用户相关SVM
        for (i,user) in enumerate(test_user):
            user_ind=np.where(test_label[:,0]==user)[0] # test库中用户记录
            user_pos_ind=np.where((test_label[:,0]==user) & (test_label[:,1]==1))[0] # test库中用户真实样本记录
            user_train_ind=np.random.choice(user_pos_ind,12,replace=False) # 随机采样24个真实签名中的12个做正样本
            user_test_ind=user_ind[~np.isin(user_ind,user_train_ind)]


            skew = neg_vecs.shape[0] / user_pos_ind.shape[0]
            svm_input=np.vstack([neg_vecs,test_vec[user_pos_ind,:]])
            svm_label=np.concatenate([np.zeros(neg_vecs.shape[0]),np.ones(user_pos_ind.shape[0])])
            svm=sklearn.svm.SVC(class_weight={1:skew},gamma=0.0048,probability=True)
            svm_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                                 ('classifier', svm)])
            svm_with_scaler.fit(svm_input,svm_label)
            hyper_dist=svm_with_scaler.decision_function(test_vec[user_test_ind,:])
            result.append(np.vstack([hyper_dist,test_label[user_test_ind,1]]))

        result=np.hstack(result).T

        fpr, tpr, thresholds = roc_curve(result[:,1],result[:,0], pos_label=1)
        fnr = 1 -tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
        pred_label=result[:,1].copy()
        pred_label[pred_label>eer_threshold]=1
        pred_label[pred_label<=eer_threshold]=0
        acc=(pred_label==result[:,1]).sum()/result.shape[0]

        area = auc(fpr, tpr)
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





