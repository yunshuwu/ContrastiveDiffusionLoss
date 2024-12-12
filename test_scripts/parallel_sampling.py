"""
In this py file, we show how to do parallel sampling (a simplified ParaSam Algo) given a pretrained diffusion model 'diffusionmodel'

Parallel sampling still use multiprocessing as parallel computation, using producer/consumer design.
We spawn a single producer process on GPU:0 to run the main loop of ParaSam, and all GPUs (including GPU:0) are as consumer processes, running parallel part of the ParaSam
"""
import argparse
import json
import re

import torch
import torch.nn as nn
import types
import torch.multiprocessing as mp

import sys
sys.path.append("..")
from diffusion_models.para_samplers import *
from diffusion_models.sampling_schedule import *
from diffusion_models.diffusionmodel import *
from network_arch.nets import *
from datasets.flow_synthetic_2Ddata import sample_2d_synthetic
from datasets.data_generator import *



class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])




def run(rank, total_ranks, queues, seeds, 
        diffusion_loss='cdl', dataset_name='dino', 
        schedule_name='ddpm', num_inference_steps=1000, batch_size=64, 
        tolerance=1e-1, mmd_threshold=1e10, 
        outdir='../test_scripts/results_ddpm_scheduler/einstein'):
    """
    Args: 
        rank: GPU:No, which GPU node
        total_ranks: number of total GPUs to be used
        queues: in total two queues for parallel computations, queues[0] for main-loop, queues[1] for parallel workers to store results
        diffusionmodel: pretrained diffusion model, we sample from its distribution
    """
    # 1. Load pretrained diffusionmodels
    # denoiser = MLP(2, 64, 3, activation=nn.ReLU, dropout=0.)
    denoiser = BasicDiscreteTimeModel(d_model=128, n_layers=2)
    print(f'Loading {diffusion_loss} loss trained {dataset_name} dataset...')

    if dataset_name.lower() == 'dino':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained Dino dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/0418_dino_epoch2000/version_0_cdl_1perbatch/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_0/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD diffusion loss trained Dino dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/0418_dino_epoch2000/version_1_itd/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_1/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM diffusion loss trained Dino dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/0418_dino_epoch2000/version_2_ddpm/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_2/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
    
    elif dataset_name.lower() == 'real_dino':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained Dino dataset...')
            ckpt_path = '../lightning_logs/version_3/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD diffusion loss trained Dino dataset...')
            ckpt_path = '../lightning_logs/version_4/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM diffusion loss trained Dino dataset...')
            ckpt_path = '../lightning_logs/version_5/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    elif dataset_name.lower() == 'einstein':
        if diffusion_loss.lower() == 'cdl':
            ckpt_path = '../lightning_logs/einstein_cdl/version_0/checkpoints/epoch=1999-step=7814000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            ckpt_path = '../lightning_logs/einstein_ddpm/version_0/checkpoints/epoch=1999-step=7814000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters']) 

    # spirals
    elif dataset_name.lower() == 'spirals':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained two-spirals dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_0/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_3/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD contrastive diffusion loss trained two-spirals dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_1/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_4/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel_loss_test(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM contrastive diffusion loss trained two-spirals dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_2/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_5/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    # moons
    elif dataset_name.lower() == 'moons':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained moons dataset...')
            ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_3/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD contrastive diffusion loss trained moons dataset...')
            ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_4/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel_loss_test(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM contrastive diffusion loss trained moons dataset...')
            ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_5/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    # eight_gaussian
    elif dataset_name.lower() == 'eight_gaussian':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained eight_gaussian dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_6/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_9/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD contrastive diffusion loss trained eight_gaussian dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_7/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_10/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel_loss_test(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM contrastive diffusion loss trained eight_gaussian dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_8/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_11/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    # checkerboard
    elif dataset_name.lower() == 'checkerboard':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained checkerboard dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_9/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_12/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD contrastive diffusion loss trained checkerboard dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_10/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_13/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel_loss_test(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM contrastive diffusion loss trained checkerboard dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_11/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_14/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    # circle
    elif dataset_name.lower() == 'circle':
        if diffusion_loss.lower() == 'cdl':
            print(f'Loading CDL contrastive diffusion loss trained circle dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_12/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_15/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'itd':
            print(f'Loading ITD contrastive diffusion loss trained circle dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_13/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_16/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = ITD_DiffusionModel_loss_test(denoiser, **ckpt['hyper_parameters'])
        elif diffusion_loss.lower() == 'ddpm':
            print(f'Loading DDPM contrastive diffusion loss trained circle dataset...')
            # ckpt_path = '../lightning_logs/lightning_logs(forgot to call dataset_info())/version_14/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt_path = '../lightning_logs/version_17/checkpoints/epoch=1999-step=782000.ckpt'
            ckpt = torch.load(ckpt_path)
            dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    dm.load_state_dict(ckpt['state_dict'])
    dm.eval()

    print(f'len(seeds) = {len(seeds)}')
    num_batches = (len(seeds) // batch_size)
    if num_batches == 0:
        all_batches = (torch.as_tensor(seeds), )
    else:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    # 1.2 If checking MMD, need ground truth dataset
    MMD_FLAG = 1e9
    if mmd_threshold < MMD_FLAG:
        if dataset_name.lower() == 'real_dino':
            print('real_dino')
            train_dl, val_dl, x_shape, train_ds = dino_dataset(n=batch_size, batch_size=128, path_2d="../datasets/assets/DatasaurusDozen.tsv")
        elif dataset_name.lower() == 'einstein':
            train_narray = create_einstein_data(n=batch_size, face='einstein')
            data_tensor = torch.from_numpy(train_narray.astype(np.float32))
            # x_shape = data_tensor.shape[1:]
            train_ds = TensorDataset(data_tensor)
        else:
            train_ds, val_ds, x_shape = sample_2d_synthetic(dataset=dataset_name, num_samples=batch_size)
        true_samples = train_ds.tensors[0]
        print(f'Get ground truth data samples.')

    num_consumers = total_ranks
    
    # 2. Get sampling schedule
    if schedule_name.lower() == 'ddpm':
        scheduler = DDPM_Scheduler(num_inference_timesteps=num_inference_steps)
    elif schedule_name.lower() == 'ddim':
        scheduler = DDIM_Scheduler(num_inference_timesteps=num_inference_steps)

    # 3. Run parallel sampling
    for batch_id, batch_seeds in enumerate(all_batches):
        cur_bz_size = len(batch_seeds)

        if rank != -1:
            # parallel GPU workers, not the main loop
            dm.to(f"cuda:{rank}")
            parasam_forward_worker(model=dm, mp_queues=queues, device=f"cuda:{rank}")
        else:
            # GPU:0 runs the main loop
            dm = dm.to(f"cuda:0")

            rnd = StackedRandomGenerator(device="cuda:0", seeds=batch_seeds)

            # warmup, do not use these outputs
            print(f'Warmup call of parasam_forward() ...')
            if mmd_threshold < MMD_FLAG: 
                _, _, _ = parasam_forward_mmd(dm, scheduler, 
                                              randn=rnd.randn,
                                              diffusionmodel_name=diffusion_loss, 
                                              num_samples=cur_bz_size, 
                                              true_samples=true_samples, # MMD check needs ground truth samples
                                              tolerance=tolerance, 
                                              mmd_threshold=mmd_threshold, # TODO: try different MMD thresholds
                                              mp_queues=queues, 
                                              device=f"cuda:0", 
                                              num_consumers=num_consumers)
            else: # No MMD calculation required
                _, _ = parasam_forward(dm, scheduler, 
                                       randn=rnd.randn,
                                       diffusionmodel_name=diffusion_loss,
                                       num_samples=cur_bz_size, 
                                       tolerance=tolerance, mp_queues=queues, 
                                       device=f"cuda:0", 
                                       num_consumers=num_consumers)
            
            # generate samples, run the main loop
            print(f'Parallel sampling ...')
            if mmd_threshold < MMD_FLAG: # with MMD threshold
                samples, stats, mmd_buffer = parasam_forward_mmd(dm, scheduler, 
                                                                 randn=rnd.randn,
                                                                 diffusionmodel_name=diffusion_loss, 
                                                                 num_samples=cur_bz_size, 
                                                                true_samples=true_samples, # MMD check needs ground truth samples
                                                                tolerance=tolerance, 
                                                                mmd_threshold=mmd_threshold, # TODO: try different MMD thresholds
                                                                mp_queues=queues, 
                                                                device=f"cuda:0", 
                                                                num_consumers=num_consumers)
            else: # w/o MMD threshold
                samples, stats = parasam_forward(dm, scheduler, 
                                                 randn=rnd.randn,
                                                 diffusionmodel_name=diffusion_loss, 
                                                 num_samples=cur_bz_size, 
                                                 tolerance=tolerance, mp_queues=queues, 
                                                 device=f"cuda:0", 
                                                 num_consumers=num_consumers)
                
            # save these generated samples, plot the 2D points shape to check if sampling is correct
            # also need to check the final mmd, comparing with the ground-truth training samples
            output_dir = './results_{}_scheduler/{}_{}'.format(schedule_name, dataset_name, diffusion_loss)
            os.makedirs(output_dir, exist_ok=True)
            
            samples_np = samples.cpu().numpy()
            sample_path = os.path.join(output_dir, f'samples_{batch_id}.pt') 
            torch.save(samples_np, sample_path)

            # store stats to file
            stats_path = os.path.join(output_dir, f'stats.json')
            with open(stats_path, 'a') as json_file:
                json.dump(stats, json_file)
                json_file.write('\n')
            
            # save the mmd buffer, for visualization MMD plot
            if mmd_threshold < MMD_FLAG:
                mmds = torch.tensor(mmd_buffer)
                mmd_path = os.path.join(output_dir, f'mmd_{batch_id}.pt')
                torch.save(mmds, mmd_path)

            # shutdown workers
            for _ in range(total_ranks):
                queues[0].put(None) 


def main(args): 
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue() # Yunshu: actually I think only need two queues, since no need to get user inputs from keyboard

    processes = []
    num_processes = torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(rank, num_processes, queues, args.seeds, args.diffusion_name, args.dataset, args.schedule_name, args.num_inference_steps, args.batch_size, args.tolerance, args.mmd_threshold, args.outdir))
        p.start()
        processes.append(p)

    # wait for all subprocesses to finish
    for p in processes:
        p.join()


def parse_int_list(s):
    """This function is from karras EDM implementation: """
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Choice
    parser.add_argument("--diffusion_name", type=str, default="cdl", choices=["cdl", "itd", "ddpm", "ddim", "edm"], help="Which diffusion model to be used.")
    # Dataset
    parser.add_argument("--dataset", type=str, default="real_dino", choices=["einstein", "dino", "real_dino", "spirals", "moons", "eight_gaussian", "checkerboard", "circle"], help="Type of the dataset to be used.")
    # number of samples to generate
    parser.add_argument("--schedule_name", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Which sampling scheduler to use.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of inference steps.")
    parser.add_argument("--batch_size", type=int, default=64, help="Maximum number of batch samples to generate from the pretrained diffusion model.")
    parser.add_argument("--seeds", type=parse_int_list, default='0-63', help="Random seeds (e.g. 1,2,5-10)")
    parser.add_argument("--tolerance", type=float, default=1e-1, help="Tolerance of parallel sampling, controls when to stop iterations by picard iteration's convergence.")
    parser.add_argument("--mmd_threshold", type=float, default=1e-4, help="MMD threshold of parallel sampling, controls when to stop iterations by MMD score. Provice 1e10 if you don't want set this control")
    parser.add_argument("--outdir", type=str, help="Path to store the images generated by parallel sampling.")

    args = parser.parse_args()

    main(args=args)