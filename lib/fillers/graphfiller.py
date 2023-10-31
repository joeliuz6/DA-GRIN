import torch
import torch.nn as nn

from . import Filler
from ..nn.models import MPGRUNet, GRINet, BiMPGRUNet

def Coral(source, target): 

    source = source.permute(0, 3, 2, 1)
    source = source.reshape(source.shape[0]*source.shape[1]*source.shape[2], source.shape[3])
    #source = source.reshape(source.shape[0]*source.shape[1],-1)

    target = target.permute(0, 3, 2, 1)
    target = target.reshape(target.shape[0]*target.shape[1]*target.shape[2], target.shape[3])
    #target = target.reshape(target.shape[0]*target.shape[1],-1)

    d = source.size(1)

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source

    #print(xm.shape)
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss
        
compute_mmd = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)



class GraphFiller(Filler):

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        super(GraphFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        if model_class is MPGRUNet:
            self.trimming = (warm_up, 0)
        elif model_class in [GRINet, BiMPGRUNet]:
            self.trimming = (warm_up, warm_up)

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):

        batch_target = batch['target']

        ##### source domain operation ######

        batch = batch['source']

        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Compute masks
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        eval_mask = batch_data.pop('eval_mask', None)
        eval_mask = (mask | eval_mask) - batch_data['mask']  # all unseen data

        y = batch_data.pop('y')

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions = (res[0], res[1]) if isinstance(res, (list, tuple)) else (res, [])

        fwd_repr_s, bwd_repr_s, imputation_repr_s = res[2],res[3],res[4]

        s_repr = torch.cat([fwd_repr_s,bwd_repr_s],dim=1)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)

        loss = self.loss_fn(imputation, target, mask)
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)


        #### target domain operation ## 
        

        # Unpack batch
        batch_data_t, batch_preprocessing_t = self._unpack_batch(batch_target)

        # Compute masks
        mask_t = batch_data_t['mask'].clone().detach()
        batch_data_t['mask'] = torch.bernoulli(mask_t.clone().detach().float() * self.keep_prob).byte()
        eval_mask_t = batch_data_t.pop('eval_mask', None)
        eval_mask_t = (mask_t | eval_mask_t) - batch_data_t['mask']  # all unseen data

        y_t = batch_data_t.pop('y')

        # Compute predictions and compute loss
        res_t = self.predict_batch(batch_target, preprocess=False, postprocess=False)
        imputation_t, predictions_t = (res_t[0], res_t[1]) if isinstance(res_t, (list, tuple)) else (res_t, [])
        fwd_repr_t,bwd_repr_t,imputation_repr_t = res_t[2],res_t[3],res_t[4]
        t_repr = torch.cat([fwd_repr_t,bwd_repr_t],dim=1)

        coral_loss = Coral(imputation_repr_s,imputation_repr_t)



        #imputation_repr_s = imputation_repr_s.permute(0, 3, 2, 1)
        #imputation_repr_s = imputation_repr_s.reshape(imputation_repr_s.shape[0]*imputation_repr_s.shape[1]*imputation_repr_s.shape[2], imputation_repr_s.shape[3])
        #imputation_repr_s = imputation_repr_s.reshape(imputation_repr_s.shape[0]*imputation_repr_s.shape[1],-1)
        #imputation_repr_t = imputation_repr_t.permute(0, 3, 2, 1)
        #print(imputation_repr_t.shape)
        #imputation_repr_t = imputation_repr_t.reshape(imputation_repr_t.shape[0]*imputation_repr_t.shape[1]*imputation_repr_t.shape[2], imputation_repr_t.shape[3])
        #imputation_repr_t = imputation_repr_t.reshape(imputation_repr_t.shape[0]*imputation_repr_t.shape[1],-1)

        #print(imputation_repr_t.shape)

        #mmd = compute_mmd(imputation_repr_s,imputation_repr_t)

        #print(mmd)
        #print(coral_loss)



        # trim to imputation horizon len
        imputation_t, mask_t, eval_mask_t, y_t = self.trim_seq(imputation_t, mask_t, eval_mask_t, y_t)
        predictions_t = self.trim_seq(*predictions_t)

        if self.scaled_target:
            target_t = self._preprocess(y_t, batch_preprocessing_t)
        else:
            target_t = y_t
            imputation_t = self._postprocess(imputation_t, batch_preprocessing_t)
            for i, _ in enumerate(predictions_t):
                predictions_t[i] = self._postprocess(predictions_t[i], batch_preprocessing_t)

        loss_t = self.loss_fn(imputation_t, target_t, mask_t)
        for pred_t in predictions_t:
            loss_t += self.tradeoff * self.loss_fn(pred_t, target_t, mask_t)
        


        loss += coral_loss

        loss += loss_t

        



        return loss

    def validation_step(self, batch, batch_idx,dataloader_idx = 1):
        
        
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        

        return val_loss

    def test_step(self, batch, batch_idx,dataloader_idx=1):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss
