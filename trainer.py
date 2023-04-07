# import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

seed = 0

def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

def trainer_synapse(config, model, snapshot_path):
    seed = config.TRAIN.SEED
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    base_lr = config.TRAIN.BASE_LR
    num_classes = config.DATASETS.Synapse.NUM_CLASSES
    batch_size = config.TRAIN.BATCH_SIZE * config.TRAIN.N_GPU
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=config.DATASETS.Synapse.ROOT_PATH, list_dir=config.DATASETS.Synapse.LIST_DIR, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[config.DATA.IMG_SIZE, config.DATA.IMG_SIZE])]))
    print("The length of train set is: {}".format(len(db_train)))

    # worker_init_fn

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    # for i, left in enumerate(trainloader):
    #     print(i)
    #     with torch.no_grad():
    #         temp = model(left).view(-1, 1, 300, 300)
    #     right.append(temp.to('cpu'))
    #     del temp
    #     torch.cuda.empty_cache()

    if config.TRAIN.N_GPU > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # print("Parameters: ",model.parameters())
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, amsgrad=True)#, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = config.TRAIN.MAX_EPOCHS
    max_iterations = config.TRAIN.MAX_EPOCHS * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    min_dice_loss = 1
    for epoch_num in iterator:
        dice_losses = []
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            dice_losses.append(np.round(loss_dice.item(),6))
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(f'epoch {epoch_num+1}/{len(iterator)} iteration {iter_num % len(trainloader)}/{len(trainloader)} : loss : {np.round(loss.item(),6)}, loss_ce: {np.round(loss_ce.item(),6)}, loss_dice: {np.round(loss_dice.item(),6)}')

            if iter_num % 20 == 0:
                image = image_batch[len(image_batch)-1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[len(image_batch)-1, ...] * 50, iter_num)
                labs = label_batch[len(image_batch)-1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        epoch_avg_dice_loss = np.average(dice_losses)
        logging.info(f'Average Dice Loss for epoch {epoch_num} : {epoch_avg_dice_loss}')

        save_interval = 1  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            if epoch_avg_dice_loss < min_dice_loss:
                logging.info(f'Dice Loss Descreased from {min_dice_loss} to {epoch_avg_dice_loss}, saving model {config.MODEL.NAME}_best.pth for epoch_num:{(epoch_num + 1)}')
                save_model_path = os.path.join(snapshot_path, f'{config.MODEL.NAME}_best.pth')
                torch.save(model.state_dict(), save_model_path)
                logging.info("save model to {}".format(save_model_path))
                min_dice_loss = np.round(epoch_avg_dice_loss,6)

        if epoch_num >= max_epoch - 1:
            if epoch_avg_dice_loss < min_dice_loss:
                save_model_path = os.path.join(snapshot_path, f'{config.MODEL.NAME}_{max_epoch}ep.pth')
                torch.save(model.state_dict(), save_model_path)
                logging.info("save model to {}".format(save_model_path))
                iterator.close()
                break

    writer.close()
    return "Training Finished!"