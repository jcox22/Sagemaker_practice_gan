# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), k)
        self.dense_layer2 = nn.Linear(k, k)
        self.dense_layer3 = nn.Linear(k, int(input_length))

    def forward(self, x):
        l1 = self.dense_layer(x)
        l2 = self.dense_layer2(F.relu(l1))
        l3 = self.dense_layer3(F.relu(l2))
        return F.sigmoid(l3)
    
class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), k)
        self.dense_layer2 = nn.Linear(k, k)
        self.dense_layer3 = nn.Linear(k, 1)

    def forward(self, x):
        l1 = self.dense_layer(x)
        l2 = self.dense_layer2(F.relu(l1))
        l3 = self.dense_layer3(F.relu(l2))
        return F.sigmoid(l3)


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.dist_backend, dist.get_world_size()
            )
            + "Current host rank is {}. Using cuda: {}. Number of gpus: {}".format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    logger.info("Loading noise dataset")
    ## My data
    logger.info("Model loaded")
    input_length = 20
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i in range(10):
            # get the inputs
            #inputs, labels = data
            #inputs, labels = inputs.to(device), labels.to(device)
            noise = torch.randint(0, 2, size=(args.batch_size, input_length)).float()
            noise = noise.to(device)
    
            # Generate examples of even real data
            true_labels = [1] * args.batch_size
            #true_data = rank0_binary.sample(16).values
            true_labels = torch.tensor(true_labels).float()
            true_labels = true_labels.to(device)
            #true_data = torch.tensor(true_data).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #outputs = model(inputs)
            #G_of_noise = generator(noise)
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
            G_of_noise = generator(noise)
            D_of_G_of_noise = discriminator(G_of_noise)
            loss = criterion(D_of_G_of_noise, true_labels)
            generator_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 2000 == 1999:  # print every 2000 mini-batches
             #   print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
              #  running_loss = 0.0

    print("Finished Training")
    return _save_model(generator, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "generator.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, metavar="BS", help="batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )

    parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())


def model_fn(model_dir):
    logger.info("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)
