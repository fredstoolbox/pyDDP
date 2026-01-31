

import torch.multiprocessing as mp
from torch.utils.data import  DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed import init_process_group, destroy_process_group
import torch

import os
import sys
from tqdm import tqdm
import logging
from model.dncnn import DnCNN
from datetime import datetime


databatch_size = 20  
accumulate_gradient_iter = 10


#set up env for DDP, use nccl backend 
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12445"
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.device = torch.device(f"cuda:{gpu_id}")
        self.gpu_id = gpu_id

        self.model = model.to(self.device) 
        self.L1_loss = torch.nn.L1Loss().to(self.device)
        self.save_every = save_every

        # wrap the model with DDP, then use the wrapped model for training
        self.ddp_model = DDP(self.model, device_ids=[gpu_id])
        self.optimizer = ZeroRedundancyOptimizer(self.ddp_model.parameters(),  optimizer_class=torch.optim.Adam, lr=0.01)
    
    #run a single batch
    def _run_batch(self, source, targets, stepindex):
        output = self.ddp_model(source)
        loss = self.L1_loss(output, targets) / accumulate_gradient_iter
        #backward is called every batch
        loss.backward()

        #only step optimizer every accumulate_gradient_iter
        if(stepindex+1) % accumulate_gradient_iter == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    #run an epoch
    def _run_epoch(self, epoch, train_data:DataLoader):
        timenow = datetime.now()
        logging.info(f"{timenow} [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(train_data)}")
        
        train_data.sampler.set_epoch(epoch) #reset sampler for each epoch
        tqdm_train_data = train_data

        #only show tqdm bar on rank 0 with tqdm, all processes are automatic sync by the DDP forward and backward pass
        if self.gpu_id == 0:
            tqdm_train_data = tqdm(train_data)

        for i, data in enumerate(tqdm_train_data, 0):
            source = data[0].to(self.device) 
            targets = data[1].to(self.device) 
            self._run_batch(source, targets, i)

        deltatime = datetime.now() - timenow
        logging.info(f'rank {self.gpu_id}, epoch time {deltatime.total_seconds()}')

    def _save_checkpoint(self, epoch):
        ckp = self.ddp_model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        logging.info(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int, databatch_size: int):
        
        torch.set_num_threads(1)
        for epoch in range(max_epochs):

            filename = '/media/fred/DATA_SSD/denoise_datasets/s256_clean_noisy_dataset1.ds'
            logging.info(f'loading {filename}')
            dataset = torch.load(filename, weights_only=False)           
            self.batch_size = databatch_size
            trainning_dataloader = DataLoader(dataset= dataset, batch_size=databatch_size, shuffle=False, sampler=DistributedSampler(dataset), num_workers=5, pin_memory=True, drop_last=False) #num_worker forced more paralleism between learning and data loading (gpu v cpu tasks). seems 1 is ok, more than 1 causes lot of memory usage (160GB)

            self._run_epoch(epoch, trainning_dataloader)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


# each spawned process contains the following func
# param rank: rank of the current process, automatically assigned by mp.spawn
# param world_size: total number of processes
def main(rank: int, world_size: int, total_epochs: int, batch_size: int, save_every: int):
    logging.basicConfig(filename='myapp.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) #log to stdout and a file

    ddp_setup(rank, world_size)
    logging.info(f'process rank {rank}')
    model = DnCNN(num_layers=17, num_features=64)
    
    if (rank == 0):
        logging.info(f'{model}')

    #instantiate the trainer for each process
    trainer = Trainer(model=model,  gpu_id=rank, save_every=save_every)
    trainer.train(total_epochs, batch_size)
    destroy_process_group()


if __name__ == "__main__":
    # setup logging to file and stdout
    databatch_size = 20
    epochs = 20
    save_every = 5

    # get the number of available GPUs
    world_size = torch.cuda.device_count()

    # spawn a number of processes, each mapped to a GPU
    # do not need to work out the rank here, mp.spawn handles it automatically
    mp.spawn(main, args=(world_size, epochs, databatch_size, save_every), nprocs=world_size, join=True) 
