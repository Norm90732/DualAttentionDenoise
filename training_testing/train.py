import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  
sys.path.insert(0, str(project_root))
import torch
from torch.utils.data import DataLoader
from dataloader import DataLoaderNoisyClean, dataset,transformStruct
import hydra
from omegaconf import DictConfig
import numpy as np
import mlflow
import logging
from pathlib import Path
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError,MeanAbsoluteError,PeakSignalNoiseRatio
#import models
from models import Unet, DualAttentionUnet

#pipeline
#https://hydra.cc/docs/plugins/optuna_sweeper/

def dataloader(cfg: DictConfig):
    generator = torch.Generator().manual_seed(20)
    processedDataSet = DataLoaderNoisyClean(dataset, transform=transformStruct)
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(processedDataSet, [cfg.train.train,# must sum to 1 (using percentages)
                                                                                                     cfg.train.validation,
                                                                                                     cfg.train.test],generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                                  pin_memory=cfg.train.pin_memory, num_workers=cfg.train.num_workers)
    validation_dataloader = DataLoader(validate_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                                       pin_memory=cfg.train.pin_memory, num_workers=cfg.train.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                                 pin_memory=cfg.train.pin_memory, num_workers=cfg.train.num_workers)

    return train_dataloader, validation_dataloader, test_dataloader

#optimize for SSIM
def optimize(cfg: DictConfig):
    train_dataloader, validation_dataloader, test_dataloader  = dataloader(cfg)


    #Starting mlflow to log
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment)
    with mlflow.start_run(run_name=f'hyperParameterOptunaSearch'):
        mlflow.log_params({
            'modelName': 'unet_denoising_attention_sweep_gaussian_noise_0.15',
            'learningRate' : cfg.optimizer.lr,
            'weightDecay' : cfg.optimizer.weight_decay,
            'batchSize': cfg.train.batch_size,
            'epochs' : cfg.train.epochs,
            'sumOrElement': cfg.model.sumOrElement
        })
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = hydra.utils.instantiate(cfg.model)
        model.to(device)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        criterion = hydra.utils.instantiate(cfg.train.loss).to(device)
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device) #maximized for in optuna
        psnr = PeakSignalNoiseRatio(data_range=1).to(device)

        #Training loop for optimize
        for epoch in range(cfg.train.epochs):
            running_loss = 0.0
            numBatches =0
            model.train()
            for noiseImg,cleanImg in (train_dataloader):
                   noiseImg, cleanImg = noiseImg.to(device), cleanImg.to(device)
                   optimizer.zero_grad()
                   output = model(noiseImg) #Forward
                   loss = criterion(output,cleanImg)
                   loss.backward()
                   optimizer.step()
                   running_loss += loss.item()
                   numBatches +=1

            averageLoss = running_loss/numBatches
            model.eval()

            valLoss = 0.0
            valBatches = 0
            #Maximizing SSIM in validation for Optuna
            ssim.reset()

            with torch.no_grad():
                for noiseImg,cleanImg in (validation_dataloader):
                    noiseImg, cleanImg = noiseImg.to(device), cleanImg.to(device)
                    output= model(noiseImg)
                    loss = criterion(output,cleanImg)
                    valLoss += loss.item()
                    valBatches += 1
                    ssim.update(output,cleanImg)
                    psnr.update(output,cleanImg)

            avgValLoss = valLoss/valBatches
            SSIM_OPT = ssim.compute()
            PSNR_OPT = psnr.compute()
            scheduler.step(avgValLoss)

            
            mlflow.log_metrics({
                'trainLoss': averageLoss,
                'valLoss': avgValLoss,
                'SSIM': SSIM_OPT.item(),
                'PSNR': PSNR_OPT.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'weightDecay': optimizer.param_groups[0]['weight_decay'],
                'batchSize': cfg.train.batch_size,
                'epochs': cfg.train.epochs,
                'sumOrElement': cfg.model.sumOrElement
                },step=epoch)

    return SSIM_OPT.item()

#To run finalModel use best_hyperparamters.yaml that outputs from optimize and populate all the config files with the correct values
def finalModel(cfg: DictConfig):
    train_dataloader, validation_dataloader, test_dataloader = dataloader(cfg)
    
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment)
    
    with mlflow.start_run(run_name="unetFinal_Attention_gaussian_0.15"):
        mlflow.log_params({
            'modelName': "unetFinal_Attention_gaussian_0.15",
            'learningRate': cfg.optimizer.lr,
            'weightDecay': cfg.optimizer.weight_decay,
            'batchSize': cfg.train.batch_size,
            'epochs': 20, #fixed value for ease.
        })
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = hydra.utils.instantiate(cfg.model)
        model.to(device)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        criterion = hydra.utils.instantiate(cfg.train.loss).to(device)
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)  # maximized for in optuna
        mse = MeanSquaredError().to(device)
        mae = MeanAbsoluteError().to(device)
        psnr = PeakSignalNoiseRatio(data_range=1).to(device)
        for epoch in range(20):
            #training phase
            running_loss = 0.0
            numBatches = 0
            model.train()
            for noiseImg, cleanImg in (train_dataloader):
                noiseImg, cleanImg = noiseImg.to(device), cleanImg.to(device)
                optimizer.zero_grad()  # Zero gradient
                output = model(noiseImg)  # Forward
                loss = criterion(output, cleanImg)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                numBatches += 1

            averageLoss = running_loss / numBatches
            
            #validation phase
            model.eval()
            valLoss = 0.0
            valBatches = 0
            ssim.reset()
            mse.reset()
            mae.reset()
            psnr.reset()
            
            with torch.no_grad():
                for noiseImg, cleanImg in (validation_dataloader):
                    noiseImg, cleanImg = noiseImg.to(device), cleanImg.to(device)
                    output = model(noiseImg)
                    loss = criterion(output, cleanImg)
                    valLoss += loss.item()
                    valBatches += 1
                    ssim.update(output, cleanImg)
                    mse.update(output, cleanImg)
                    mae.update(output, cleanImg)
                    psnr.update(output, cleanImg)

            avgValLoss = valLoss / valBatches
            valSSIM = ssim.compute()
            valMSE = mse.compute()
            valMAE = mae.compute()
            valPSNR = psnr.compute()
            scheduler.step(avgValLoss)
            mlflow.log_metrics({
                'trainLoss': averageLoss,
                'valLoss': avgValLoss,
                'valSSIM': valSSIM.item(),
                'valMSE': valMSE.item(),
                'valMAE': valMAE.item(),
                'valPSNR': valPSNR.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'weightDecay': optimizer.param_groups[0]['weight_decay'],
            }, step=epoch)
            

        #testing phase
        model.eval()
        testLoss = 0.0
        testBatches = 0
        ssim.reset()
        mse.reset()
        mae.reset()
        psnr.reset()
        with torch.no_grad():
            for noiseImg, cleanImg in (test_dataloader):
                noiseImg, cleanImg = noiseImg.to(device), cleanImg.to(device)
                output= model(noiseImg)
                loss = criterion(output,cleanImg)
                testLoss += loss.item()
                testBatches +=1
                ssim.update(output,cleanImg)
                mse.update(output, cleanImg)
                mae.update(output, cleanImg)
                psnr.update(output, cleanImg)

        testFinalLoss = testLoss / testBatches
        testFinalSSIM = ssim.compute()
        testFinalMSE = mse.compute()
        testFinalMAE = mae.compute()
        testFinalPSNR = psnr.compute()

        mlflow.log_metrics({
            'testFinal_loss': testFinalLoss,
            'test_final_SSIM': testFinalSSIM.item(),
            'test_final_MSE': testFinalMSE.item(),
            'test_final_MAE': testFinalMAE.item(),
            'test_final_PSNR': testFinalPSNR.item()

        })
        
        
        model_path = Path(cfg.logger.save_dir) / f"unetFinal_Attention_gaussian_0.15.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))


#Main Function to either run sweep or final training with config files to populate.
@hydra.main(config_path='/blue/uf-dsi/normansmith/projects/TVAttention-Autoencoder/configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.get('mode','optimize') == 'optimize':
        metric = optimize(cfg)
        return metric
    elif cfg.get('mode','final') == 'final':
        return finalModel(cfg)


if __name__ == "__main__":
    main()














