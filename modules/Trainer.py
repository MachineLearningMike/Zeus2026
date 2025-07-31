
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table

import env
import recorder
import metrics
from Zeus import Zeus

class Trainer():

    def __init__(self, config):
        self.image_height = config['height']                        # 720
        self.image_width = config['width']                          # 1,440
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len
        self.nFramesOut = 1 * config['FG']                          # 24

        self.patch_size = config['patch_size']                      # 16
        self.nPatches_h = self.image_height // self.patch_size      # 720 / 16 = 45
        self.nPatches_w = self.image_width // self.patch_size       # 1,440 / 16 = 90
        self.nPatches = self.nPatches_h * self.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
        self.nSequences = self.nFramesIn * self.nPatches            # 48 * 4,050 = 194,400

        self.nChannels = config['channels']                         # 5
        self.patch_dim = self.nChannels * self.patch_size ** 2      # 16 * 16 * 5 = 1,280
        self.dim = self.patch_dim
        
        self.nClasses = self.nChannels
        self.nHeads = config['heads']
        self.dim_head = config['dim_head']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.drop_path = config['drop_path']
        self.scale_dim = config['scale_dim']

        self.model = Zeus(config)

    def validate(self, data_loader, steps):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(data_loader)
        data_loader.prefetch(steps)     # Fetch one batch asynchronously
        
        for batch_x, batch_y in vali_pbar:  # Wait for the asynchronous .prefetch
            assert tuple(batch_y.shape)[1] == steps * self.config('FG')    # (B,steps*SG,C,H,W)
            data_loader.prefetch(steps)     # Fetch one batch asynchronously

            batch_x, batch_y = batch_x.to(env.device), batch_y.to(env.device)
            pred_list = []

            for step in range(steps):
                pred_y = self.model(batch_x, pred_list)   # (B,SG,C,H,W). t is implicit.
                pred_list.append(pred_y)

            pred_y = torch.cat(pred_list, dim=1)   # (B,steps*SG,C,H,W)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                    pred_y, batch_y], [preds_lst, trues_lst]))      # lst = [ (B,steps*SG,C,H,W) ] * n

            loss = env.criterion(pred_y, batch_y)
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)   # (T,B,C,H,W)
        trues = np.concatenate(trues_lst, axis=0)   # (T,B,C,H,W)
        mse, mae, t_mae, ssim, psnr = metrics.metric(preds, trues, data_loader.dataset.mean, data_loader.dataset.std, True)
        #self.t_sample = t_mae/np.sum(t_mae) <- only need for long epoch

        env.print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        
        return total_loss

def _conclude_epoch(self, tape_recorder, epoch, train_loss, args):
    train_loss = np.average(train_loss)

    with torch.no_grad():
        vali_loss = self.validate(self.model, env.vali_loader)
        if epoch % (args.log_step * 100) == 0: env._save(name=str(epoch))
        
    env.print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
        epoch + 1, train_loss, vali_loss))
    tape_recorder(vali_loss, self.model, env.path)
    go_next_epoch = True
    
    return go_next_epoch

def train(self, data_loader, args):
    config = args.__dict__
    tape_recorder = recorder.Recorder(verbose=True)

    for epoch in range(config['epochs']):
        self.model.train()
        train_loss = []
        pbar = tqdm(env.train_loader)

        new_T2 = env.get_next_steps()
        data_loader.prefetch(new_T2)        # Fetch one batch asynchronously

        for batch_x, batch_y in pbar:         # Wait for the asynchronous .prefetch
            torch.cuda.synchronize()
            # batch_x: (B,T1*SG,C,H,W), batch_y: (B,T2*SG,C,H,W), T2 = new_steps
            T2 = new_T2; assert tuple(batch_y.shape)[1] == T2 * self.config('FG')
            new_T2 = env.get_next_steps(); data_loader.prefetch(new_T2)    # Fetch one batch asynchronously

            batch_x, batch_y = batch_x.to(env.device), batch_y.to(env.device)
            for step in range(T2):
                pred_y = self.model(batch_x)     # (B,SG,C,H,W). t is implicit.
                target = batch_y[:, step*self.config('FG') : (step+1)*self.config('FG'), :, :, :]
                loss = env.criterion(pred_y, target)
                loss.append(loss.item())
                pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                env.optimizer.zero_grad()
                loss.backward()             
                env.optimizer.step()

            env.scheduler.step()

            if env.step % env.update_ema_every == 0: env.step_ema()
            env.step += 1

        go_next_epoch = _conclude_epoch(tape_recorder, epoch, train_loss, args)
        if not go_next_epoch: break

    best_model_path = env.path + '/' + 'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))
    return self.model

def test(self, model, data_loader):
    model.eval()
    preds_lst, trues_lst, total_loss = [], [], []
    vali_pbar = tqdm(data_loader)

    new_steps = env.get_next_steps()
    data_loader.prefetch(new_steps)        # Fetch one batch asynchronously
    
    for batch_x, batch_y in vali_pbar:         # Wait for the asynchronous .prefetch
        curr_steps = new_steps
        assert tuple(batch_y.shape)[1] == curr_steps * self.config('FG')   # (B,T2,C,H,W), T2 = curr_steps 
        new_steps = env.get_next_steps()
        data_loader.prefetch(new_steps)    # Fetch one batch asynchronously

        pred_list = []
        for step in range(curr_steps):
            pred_y = model(batch_x.to(env.device), pred_list)    # (B,SG,C,H,W). t is implicit.
            pred_list.append(pred_y)

        pred_y = torch.cat(pred_list, dim=1).unsqueeze(2)
        list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                pred_y, batch_y], [preds_lst, trues_lst]))  # lst = [ (B,steps*SG,C,H,W) ] * n

        loss = env.criterion(pred_y, batch_y)
        vali_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
        total_loss.append(loss.mean().item())

    total_loss = np.average(total_loss)
    preds = np.concatenate(preds_lst, axis=0)
    trues = np.concatenate(trues_lst, axis=0)
    mse, mae, t_mae, ssim, psnr = metrics.metric(preds, trues, data_loader.dataset.mean, data_loader.dataset.std, True)
    #self.t_sample = t_mae/np.sum(t_mae) <- only need for long epoch

    env.print_log('test mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
    
    return total_loss

# def inference(model, lat, lon, start_hour, hours):
#     # sketch only
#     x, t_date = get_latest_truth_data()     # x: (T1,C,H,W)
#     steps = get_steps(t_date, start_hour, hours)

#     batch_x = x.unsqueeze(0)    # (B=1,T1,C,H,W)

#     pred_list = []
#     for step in range(steps):
#         pred_y = model(batch_x.to(env.device), pred_list)    # (B,SG,C,H,W). t is implicit.
#         pred_list.append(pred_y)
    
#     response = get_response(pred_list, t_date, lat, lon, start_hour, hours)
#     response = format_zeuse(response)

#     return response

if __name__ == "__main__":
    config = {
        # image h w c
        'height': 720,
        'width': 1440,
        'channels': 5,
        # video length in and out
        'T1': 2,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 16,
        # transformer
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
    }

    model = Zeus(config)