import torch.nn as nn
import torch


class ClusteringAffinity(nn.Module):
    def __init__(self,n_classes,n_centers,sigma,feat_dim,init_weight=True,**kwargs):
        super(ClusteringAffinity,self).__init__()
        self.n_classes=n_classes
        self.n_centers=n_centers
        self.feat_dim=feat_dim
        self.sigma=sigma
        self.centers=nn.Parameter(torch.randn(self.n_classes,self.n_centers,self.feat_dim))
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self,f):
        f_expand=f.unsqueeze(1)
        w_expand=self.centers.unsqueeze(0)
        fw_norm=torch.sum((f_expand-w_expand)**2,-1)
        distance=torch.exp(-fw_norm/self.sigma)
        distance=torch.max(distance,-1)

        #Regularization
        mc=self.n_centers*self.n_classes
        w_reshape=self.centers.view(mc,self.feat_dim)
        w_reshape_expand1=w_reshape.unsqueeze(0)
        w_reshape_expand2=w_reshape.unsqueeze(1)
        w_norm_mat=torch.sum((w_reshape_expand2-w_reshape_expand1)**2,-1)
        w_norm_upper=torch.triu(w_norm_mat)
        mu=2.0/(mc**2-mc)*w_norm_upper.sum()
        residuals=((w_norm_upper-mu)**2).triu()
        rw=2.0/(mc**2-mc)*residuals.sum()
        batch_size=f.size(0)
        rw_broadcast=torch.ones((batch_size,1))*rw
        output=torch.cat((distance,rw_broadcast),-1)
        return output


class Affinity_Loss(nn.Module):
    def __init__(self,lambd):
        super(Affinity_Loss,self).__init__()
        self.lamda=lambd

    def forward(self, y_tru_pluseone,y_pred_plusone):
        onehot=y_tru_pluseone[:,:-1]
        distance=y_pred_plusone[:,:-1]
        rw=torch.mean(y_pred_plusone[:,-1])
        d_fi_wyi=torch.sum(onehot*distance,-1,keepdims=True)
        losses=torch.clam(self.lamda+distance-d_fi_wyi,min=0)
        L_mm=torch.sum(losses*(1-onehot),-1)
        return L_mm+rw
