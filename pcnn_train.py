import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode='training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()

    deno = args.batch_size * np.prod(args.obs) * np.log(2.)
    loss_tracker = mean_tracker()

    # Optionally, if evaluating accuracy:
    if mode != 'training':
        num_correct = 0
        num_total = 0

    for batch_idx, (model_input, categories) in enumerate(tqdm(data_loader)):
        model_input = model_input.to(device)
        label_int = [my_bidict[category] for category in categories]
        label_tensor = torch.tensor(label_int, dtype=torch.long).to(device)
        
        model_output = model(model_input, class_label=label_tensor)
        loss = loss_op(model_input, model_output)
        loss_tracker.update(loss.item() / deno)

        if mode != 'training':
            predicted_labels = get_label(model, model_input, device)
            num_correct += (predicted_labels == label_tensor).sum().item()
            num_total += model_input.size(0)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if mode != 'training':
        accuracy = num_correct / num_total
        print(f"{mode.capitalize()} Accuracy: {accuracy:.2%}")
        wandb.log({mode + "-Accuracy": accuracy})
    wandb.log({mode + "-Average-BPD": loss_tracker.get_mean(), mode + "-epoch": epoch})

def get_label(model, model_input, device):
    model.eval()
    batch_size = model_input.size(0)
    class_losses = []
    for i in range(len(my_bidict)):
        label_tensor = torch.full((batch_size,), i, dtype=torch.long, device=device)
        out = model(model_input, sample=False, class_label=label_tensor)
        loss_per_sample = discretized_mix_logistic_loss(model_input, out, reduce=False)
        class_losses.append(loss_per_sample)
    class_losses = torch.stack(class_losses)
    predicted_labels = torch.argmin(class_losses, dim=0)
    return predicted_labels

def main(args):
    # Initialize wandb (will use sweep config values if sweep run)
    wandb.init(project="CPEN455HW")
    
    # Here, if you wish, override your args with wandb.config:
    args.batch_size = wandb.config.batch_size
    args.lr_decay = wandb.config.lr_decay
    args.max_epochs = wandb.config.max_epochs
    args.nr_filters = wandb.config.nr_filters
    args.nr_logistic_mix = wandb.config.nr_logistic_mix
    args.nr_resnet = wandb.config.nr_resnet

    pprint(args.__dict__)

    check_dir_and_create(args.save_dir)

    # Set reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_" + ("load_model" if args.load_params else "from_scratch")
    model_path = os.path.join(args.save_dir, model_name)
    
    job_name = "PCNN_Training_dataset:" + args.dataset + "_" + args.tag

    wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}

    # Set up data
    if args.dataset == "cpen455":
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, mode='train', transform=ds_transforms),
                                                   batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, mode='validation', transform=ds_transforms),
                                                 batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise Exception("This sweep is intended for the cpen455 dataset.")

    args.obs = (3, 32, 32)
    input_channels = args.obs[0]

    loss_op = lambda real, fake: discretized_mix_logistic_loss(real, fake, reduce=True)
    sample_op = lambda x: sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                     input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    for epoch in range(args.max_epochs):
        train_or_test(model=model, data_loader=train_loader, optimizer=optimizer,
                      loss_op=loss_op, device=device, args=args, epoch=epoch, mode='training')
        scheduler.step()

        train_or_test(model=model, data_loader=val_loader, optimizer=optimizer,
                      loss_op=loss_op, device=device, args=args, epoch=epoch, mode='val')

        # Sample and log FID every sampling interval
        if epoch % args.sampling_interval == 0:
            print('......sampling......')
            # Generate samples for each class
            for label in my_bidict.keys():
                label_tensor = torch.full((args.sample_batch_size,), my_bidict[label], dtype=torch.long, device=device)
                sample_t = sample(model, args.sample_batch_size, args.obs, sample_op, label_tensor)
                sample_t = rescaling_inv(sample_t)
                save_images(sample_t, args.sample_dir, label=label)
                sample_result = wandb.Image(sample_t, caption="epoch {} label {}".format(epoch, label))
                wandb.log({"samples_{}".format(label): sample_result})
            # Compute FID over generated samples and a reference set
            gen_data_dir = args.sample_dir
            ref_data_dir = os.path.join(args.data_dir, 'test')
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension 192 works! FID score: {}".format(fid_score))
            except Exception as e:
                print("FID calculation failed:", e)
            wandb.log({"FID": fid_score, "epoch": epoch})

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), 'models/conditional_pixelcnn.pth')
            torch.save(model.state_dict(), f'models/{model_name}_{epoch}.pth')
            wandb.save(f'models/{model_name}_{epoch}.pth')

if __name__ == '__main__':
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()

    # Add all your args here
    parser.add_argument('-w', '--en_wandb', type=bool, default=False, help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default', help='Tag for this run')

    parser.add_argument('-c', '--sampling_interval', type=int, default=5)
    parser.add_argument('-i', '--data_dir', type=str, default='data')
    parser.add_argument('-o', '--save_dir', type=str, default='models')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples')
    parser.add_argument('-d', '--dataset', type=str, default='cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10)
    parser.add_argument('-r', '--load_params', type=str, default=None)
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32))

    parser.add_argument('-q', '--nr_resnet', type=int, default=1)
    parser.add_argument('-n', '--nr_filters', type=int, default=40)
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5)
    parser.add_argument('-l', '--lr', type=float, default=0.0002)
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32)
    parser.add_argument('-x', '--max_epochs', type=int, default=5000)
    parser.add_argument('-s', '--seed', type=int, default=1)

    args = parser.parse_args()
    pprint(args.__dict__)

    # Call main with the parsed args
    main(args)