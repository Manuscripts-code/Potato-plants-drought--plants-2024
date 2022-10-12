import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def visualize(images, output):
    def imshow(img):
        # img = img / 2 + 0.5  
        plt.imshow(np.transpose(img.to("cpu"), (1, 2, 0))) 
    no_images = images.shape[0]

    #Original Images
    print("Original Images")
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(no_images):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
    # plt.show()

    #Reconstructed Images
    print('Reconstructed Images')
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(no_images):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(output[idx])
    plt.show() 


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['data_sampler'],
        config['data_loader']['args']['grouped_labels_filepath'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(os.path.join(config.resume,"model_best.pth"))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    target_list = []
    pred_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            target = target.view(len(target), -1).float()

            decoded, _, pred_class = model(data)
            loss1 = loss_fn(decoded, data) 
            loss2 = loss_fn(pred_class, target)
            loss = loss1 + loss2
            target_list.append(target)
            pred_list.append(pred_class)

            #
            # save sample images, or do something with output here
            # visualize(data[:,[55,41,12],:,:], decoded[:,[55,41,12],:,:])
            #

            # computing loss, metrics on test set
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(pred_class, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    for t, p in zip(target_list, pred_list):
        t_ = t[0][0].cpu().numpy()
        p_ = round(p[0][0].cpu().numpy().round())
        print(t_,p_)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
