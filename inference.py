import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

from .setup import device
from .config import n_classes, n_features
from .utils import show_tensor_images
from .phase2 import dataloader2, encoder, generator2

# Encode features by class label
features = {}
for (x, _, inst, _) in tqdm(dataloader2):
    x = x.to(device)
    inst = inst.to(device)
    area = inst.size(2) * inst.size(3)

    # Get pooled feature map
    with torch.no_grad():
        feature_map = encoder(x, inst)

    for i in torch.unique(inst):
        label = i if i < 1000 else i // 1000
        label = int(label.flatten(0).item())

        # All indices should have same feature per class from pooling
        idx = torch.nonzero(inst == i, as_tuple=False)
        n_inst = idx.size(0)
        idx = idx[0, :]

        # Retrieve corresponding encoded feature
        feature = feature_map[idx[0], :, idx[2], idx[3]].unsqueeze(0)

        # Compute rate of feature appearance (in official code, they compute per block)
        block_size = 32
        rate_per_block = 32 * n_inst / area
        rate = torch.ones((1, 1), device=device).to(feature.dtype) * rate_per_block

        feature = torch.cat((feature, rate), dim=1)
        if label in features.keys():
            features[label] = torch.cat((features[label], feature), dim=0)
        else:
            features[label] = feature


# Cluster features by class label
k = 10
centroids = {}
for label in range(n_classes):
    if label not in features.keys():
        continue
    feature = features[label]

    # Thresholding by 0.5 isn't mentioned in the paper, but is present in the
    # official code repository, probably so that only frequent features are clustered
    feature = feature[feature[:, -1] > 0.5, :-1].cpu().numpy()

    if feature.shape[0]:
        n_clusters = min(feature.shape[0], k)
        kmeans = KMeans(n_clusters=n_clusters).fit(feature)
        centroids[label] = kmeans.cluster_centers_

def infer(label_map, instance_map, boundary_map):
    # Sample feature vector centroids
    b, _, h, w = label_map.shape
    feature_map = torch.zeros((b, n_features, h, w), device=device).to(label_map.dtype)

    for i in torch.unique(instance_map):
        label = i if i < 1000 else i // 1000
        label = int(label.flatten(0).item())

        if label in centroids.keys():
            centroid_idx = random.randint(0, centroids[label].shape[0] - 1)
            idx = torch.nonzero(instance_map == int(i), as_tuple=False)

            feature = torch.from_numpy(centroids[label][centroid_idx, :]).to(device)
            feature_map[idx[:, 0], :, idx[:, 2], idx[:, 3]] = feature

    with torch.no_grad():
        x_fake = generator2(torch.cat((label_map, boundary_map, feature_map), dim=1))
    return x_fake

for x, labels, insts, bounds in dataloader2:
    x_fake = infer(labels.to(device), insts.to(device), bounds.to(device))
    show_tensor_images(x_fake.to(x.dtype))
    show_tensor_images(x)
    break