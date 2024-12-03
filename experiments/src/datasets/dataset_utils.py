from typing import Sequence

from glob import glob
import numpy as np
import time
import json
import os
import copy
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from datasets import PathDataset, dataset_factory
from utils import io_utils
from utils import read_txt, save_parameters_to_yaml
from clustering import Node, Link
from clustering import get_clustering
import globals

import torch
from tqdm import tqdm


def merge_datasets(datasets: Sequence[PathDataset]):
    ids = []
    images = []
    targets = []
    height = None
    width = None

    def _merge(d: Sequence[PathDataset], ids, images, targets):
        nonlocal height
        nonlocal width
        if isinstance(d, list):
            for d_i in d:
                _merge(d_i, ids, images, targets)
        elif isinstance(d, PathDataset):
            ids.extend(d.ids)
            images.extend(d.paths)
            targets.extend(d.targets)
            if height is None:
                height = d.height
            if width is None:
                width = d.width

    for dataset in datasets:
        _merge(dataset, ids, images, targets)
    return PathDataset(ids, images, targets, height, width)

def extract_tensor_dataset(dataset,
                           setting_strategy,
                           streams,
                           tensor_dataset_folders,
                           folder,
                           num_workers=4,
                           pin_memory=True,
                           persistent_workers=False
                           ):
    with torch.no_grad():
        batch_size = (256 // len(streams)) * len(streams)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers
                                )  

        for mini_batch in tqdm(dataloader):  #(id, [images], targets, task)
            input = setting_strategy.before_feature_extraction(mini_batch).to(
                setting_strategy.model.device, non_blocking=True)
                                    
            freezed_f = setting_strategy.extract_freezed_features(input)
            freezed_f = setting_strategy.after_freezed_features_extraction(freezed_f)
            freezed_f = freezed_f.detach().cpu()
            
            setting_strategy.save_features(freezed_f, 
                                            ids=mini_batch[0], 
                                            tensor_dataset_folders=tensor_dataset_folders,
                                            folder=folder)
            
    
def _load_data_for_clustering(sequences_paths: Sequence[str],
                              image_folders: Sequence[str],
                              poses_files: Sequence[str],
                              dataset_type: str,
                              nodes_files: Sequence[str] = None,
                              links_files: Sequence[str] = None):

    nodes = []
    links = []
    dataset = dataset_factory(dataset_type)
    id_counter = 0
    for i in range(len(sequences_paths)):
        current_images_paths = []
        current_weights = None
        current_timestamps = None
        current_links = None
        id_map = None
        for j in range(len(image_folders)):
            paths = glob(
                os.path.join(sequences_paths[i], image_folders[j], "*"))
            paths.sort()
            current_images_paths.append(paths)

        current_ids = np.arange(id_counter,
                                id_counter + len(current_images_paths[0]))

        id_counter += len(current_ids)

        current_images_paths = list(zip(*current_images_paths))

        poses = dataset.read_file(
            os.path.join(sequences_paths[i], poses_files[i]))
        current_positions = dataset.get_translations(poses)

        try:
            current_timestamps = dataset.get_timestamps(poses)
        except:
            pass

        if nodes_files is not None:
            ids_weights = read_txt(os.path.join(sequences_paths[i],
                                                nodes_files[i]),
                                   header=0)
            current_weights = ids_weights[:, 1]
            nodes_ids = ids_weights[:, 0]
            id_map = dict()
            for j, id in enumerate(nodes_ids):
                id_map[id] = current_ids[j]

        if links_files is not None:
            current_links = read_txt(os.path.join(sequences_paths[i],
                                                  links_files[i]),
                                     header=0)
            for j in range(len(current_links)):
                current_links[j, 0] = id_map[current_links[j, 0]]
                current_links[j, 1] = id_map[current_links[j, 1]]

        current_nodes = []
        for j, id in enumerate(current_ids):
            current_nodes.append(
                Node(
                    id,
                    i,
                    current_images_paths[j],
                    current_weights[j]
                    if current_weights is not None else None,
                    current_positions[j],
                    current_timestamps[j]
                    if current_timestamps is not None else None,
                ))

        current_links = [
            Link(current_links[j][0], current_links[j][1], current_links[j][2])
            for j in range(len(current_links))
        ] if current_links is not None else None

        nodes.append(current_nodes)
        if current_links is not None:
            links.append(current_links)

    return nodes, links if len(links) > 0 else None


def _compute_dynamic_train_dataset(json_dataset, experience_size,
                                   setting_strategy, image_height,
                                   image_width):
    train_dataset = []
    nodes_moved_in_each_experience = []
    for i in range(len(json_dataset)):  #for each sequence
        train_dataset.append([])
        start = 0
        end = experience_size
        while start < end:
            ids = []
            images = []
            targets = []
            nodes_moved_in_this_experience = dict()
            current_exp = json_dataset[i][start:end]
            for data in current_exp:
                ids.append(data[0])
                images.append(data[1])
                targets.append(data[2])
                for key, value in data[3].items():
                    k = int(key)
                    if k in nodes_moved_in_this_experience:
                        if nodes_moved_in_this_experience[k][0] == value[
                                1]:  #reassigned back to the same cluster
                            nodes_moved_in_this_experience.pop(k)
                        else:
                            nodes_moved_in_this_experience[k] = (
                                nodes_moved_in_this_experience[k][0], value[1]
                            )  #reassigned to a different cluster
                    else:
                        nodes_moved_in_this_experience[
                            k] = value  #assigned to a new cluster

            ids, images, targets = setting_strategy.prepare_data(
                ids, images, targets)

            train_dataset[i].append(
                PathDataset(copy.deepcopy(ids), copy.deepcopy(images),
                            copy.deepcopy(targets), image_height, image_width))
            nodes_moved_in_each_experience.append(
                copy.deepcopy(nodes_moved_in_this_experience))

            start += experience_size
            end = len(json_dataset[i]) if end + experience_size > len(
                json_dataset[i]) else end + experience_size
    return train_dataset, nodes_moved_in_each_experience


def _compute_end_dataset(json_dataset, setting_strategy, image_height,
                         image_width):
    ids, images, targets = setting_strategy.prepare_data(
        json_dataset[0], json_dataset[1], json_dataset[2])
    return PathDataset(ids, images, targets, image_height, image_width)


def _compute_test_dataset(json_dataset, setting_strategy, image_height,
                          image_width):
    test_dataset = []
    for i in range(len(json_dataset)):  #for each sequence
        ids = []
        images = []
        targets = []
        for j in range(len(json_dataset[i])):  #for each example
            ids.append(json_dataset[i][j][0])
            images.append(json_dataset[i][j][1])
            targets.append(json_dataset[i][j][2])
        ids, images, targets = setting_strategy.prepare_data(
            ids, images, targets)

        test_dataset.append([
            PathDataset(copy.deepcopy(ids), copy.deepcopy(images),
                        copy.deepcopy(targets), image_height, image_width)
        ])
    return test_dataset


def _load_dataset(dataset_folder, dataset_file, load_fn):
    path = os.path.join(dataset_folder, dataset_file)
    if os.path.exists(path):
        with open(path, 'r') as json_file:
            json_dataset = json.load(json_file)
            loaded_data = load_fn(json_dataset)
        return loaded_data
    return None


def _create_dataset(dataset_folder, image_folders, train_paths,
                    train_poses_files, train_nodes_files, train_links_files,
                    clustering_params, image_height, image_width,
                    end_train_dataset_file, dynamic_train_dataset_file,
                    experience_size, use_validation, seed, balanced_validation,
                    validation_size, end_valid_dataset_file, test_paths,
                    test_poses_files, test_nodes_files,
                    test_links_files, test_dataset_file, setting_strategy):

    dataset_type = clustering_params["dataset_type"]
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    save_parameters_to_yaml(os.path.join(dataset_folder, "dataset.yaml"),
                            clustering_params)

    train_nodes, train_links = _load_data_for_clustering(
        train_paths, image_folders, train_poses_files, dataset_type,
        train_nodes_files, train_links_files)

    clustering = get_clustering(clustering_params["type"], **clustering_params)

    train_dataset_to_save = []

    if globals.debug:
        with open(globals.clustering_file, "a") as f:
            f.write("Clustering started\n")
    print("Clustering started\n")
    for i in range(len(train_nodes)):
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Clustering sequence {i+1}\n")
        print(f"Clustering sequence {i+1}")
        start_time = time.time()
        train_dataset_to_save.append([])

        #clustering
        clustering.set_current_sequence(
            train_nodes[i],
            train_links[i] if train_links is not None else None)
        print(len(train_nodes[i]))
        while clustering.clusterize_next_node(clustering_params["dynamic"]):
            current_node = clustering.nodes[clustering.current_node_idx - 1]
            current_data = [
                int(current_node.id), current_node.images,
                int(current_node.region_id)
            ]
            moved_nodes = copy.deepcopy(clustering.just_moved)
            if current_node.id in moved_nodes:
                moved_nodes.pop(current_node.id)
            moved_nodes_encoded = {}
            for key, value in moved_nodes.items():
                moved_nodes_encoded[int(key)] = (int(value[0]), int(value[1]))
            current_data.append(moved_nodes_encoded)
            train_dataset_to_save[-1].append(current_data)

        end_time = time.time()
        clustering.update_when_sequence_ended()
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Time elapsed: {end_time - start_time}\n")

    with open(os.path.join(dataset_folder, dynamic_train_dataset_file),
              'w') as json_file:
        json.dump(train_dataset_to_save, json_file)

    dynamic_train_dataset, nodes_moved_in_each_experience = _compute_dynamic_train_dataset(
        train_dataset_to_save, experience_size, setting_strategy, image_height,
        image_width)

    ids_train = []
    images_train = []
    targets_train = []
    for id, node in clustering.memory.nodes.items():
        # f = open(
        #     os.path.join(dataset_folder, f"regions_{node.sequence_id}.txt"),
        #     "a")
        # f.write(f"{node.id} {node.region_id}\n")
        # f.close()
        # f = open(
        #     os.path.join(dataset_folder, f"positions_{node.sequence_id}.txt"),
        #     "a")
        # f.write(f"{node.id} {node.position[0]} {node.position[1]}\n")
        # f.close()
        ids_train.append(int(id))
        images_train.append(node.images)
        targets_train.append(int(node.region_id))

    if use_validation:
        ids_train, \
        ids_valid, \
        images_train, \
        images_valid, \
        targets_train, \
        targets_valid = train_test_split(np.asarray(ids_train),
                        np.asarray(images_train),
                        np.asarray(targets_train),
                        test_size=validation_size,
                        random_state=seed,
                        stratify=targets_train if balanced_validation else None)
        end_valid_to_save = [
            ids_valid.tolist(),
            images_valid.tolist(),
            targets_valid.tolist()
        ]
        end_valid_dataset = _compute_end_dataset(end_valid_to_save,
                                                 setting_strategy,
                                                 image_height, image_width)
        with open(os.path.join(dataset_folder, end_valid_dataset_file),
                  'w') as json_file:
            json.dump(end_valid_to_save, json_file)

        ids_train = ids_train.tolist()
        images_train = images_train.tolist()
        targets_train = targets_train.tolist()

    end_train_to_save = [ids_train, images_train, targets_train]
    end_train_dataset = _compute_end_dataset(end_train_to_save,
                                             setting_strategy, image_height,
                                             image_width)
    with open(os.path.join(dataset_folder, end_train_dataset_file),
              'w') as json_file:
        json.dump(end_train_to_save, json_file)

    test_nodes, test_links = _load_data_for_clustering(
        test_paths, image_folders, test_poses_files, dataset_type,
        test_nodes_files, test_links_files)

    test_dataset_to_save = []
    for i in range(len(test_nodes)):
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Aligning test sequence {i+1}\n")
        print(f"Aligning test sequence {i+1}")
        start_time = time.time()
        test_dataset_to_save.append([])

        #clustering
        clustering.set_current_sequence(
            test_nodes[i], test_links[i] if test_links is not None else None)
        clustering.align_sequence()

        for node in clustering.nodes:
            current_data = [int(node.id), node.images, int(node.region_id)]
            test_dataset_to_save[-1].append(current_data)
            # f = open(
            #     os.path.join(dataset_folder,
            #                  f"test_regions_{node.sequence_id}.txt"), "a")
            # f.write(f"{node.id} {node.region_id}\n")
            # f.close()
            # f = open(
            #     os.path.join(dataset_folder,
            #                  f"test_positions_{node.sequence_id}.txt"), "a")
            # # f.write(f"{node.id} {node.position[0]} {node.position[1]} {node.position[2]}\n")
            # f.write(f"{node.id} {node.position[0]} {node.position[1]}\n")
            # f.close()

        end_time = time.time()
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Time elapsed: {end_time - start_time}\n")

    with open(os.path.join(dataset_folder, test_dataset_file),
              'w') as json_file:
        json.dump(test_dataset_to_save, json_file)

    if globals.debug:
        with open(globals.clustering_file, "a") as f:
            f.write("Aligning ended\n")
    print("Aligning ended")

    test_dataset = _compute_test_dataset(test_dataset_to_save,
                                         setting_strategy, image_height,
                                         image_width)

    return dynamic_train_dataset, \
        nodes_moved_in_each_experience,  \
        end_train_dataset, \
        end_valid_dataset, \
        test_dataset


def get_dataset(params,
                setting_strategy,
                continual=False,
                dynamic_train_dataset_file=io_utils.dynamic_train_dataset_file,
                end_train_dataset_file=io_utils.end_train_dataset_file,
                end_valid_dataset_file=io_utils.end_valid_dataset_file,
                test_dataset_file=io_utils.test_dataset_file):
    seed = params["seed"]
    dataset_params = params["dataset"]
    image_width = dataset_params["image_width"]
    image_height = dataset_params["image_height"]
    train_params = dataset_params["train"]
    test_params = dataset_params["test"]
    #clustering_type not needed
    use_validation = dataset_params["use_validation"]
    if use_validation:
        validation_params = dataset_params["validation"]

    clustering_params = params["clustering"]

    train_paths = train_params["sequences"]
    test_paths = test_params["sequences"]

    image_folders = dataset_params["images_folders"]

    train_nodes_files = train_params[
        "nodes_files"] if "nodes_files" in train_params else None
    test_nodes_files = test_params[
        "nodes_files"] if "nodes_files" in test_params else None
    train_links_files = train_params[
        "links_files"] if "links_files" in train_params else None
    test_links_files = test_params[
        "links_files"] if "links_files" in test_params else None
    train_poses_files = train_params["poses_files"]
    test_poses_files = test_params["poses_files"]

    experience_size = train_params["experience_size"]

    if dataset_params["load_dataset"]:          
        dataset_folder = dataset_params["dataset_folder"]

        dynamic_train_dataset, \
        nodes_moved_in_each_experience = _load_dataset(dataset_folder,
                                                      dynamic_train_dataset_file,
                                                      lambda json_dataset: _compute_dynamic_train_dataset(json_dataset,
                                                                                              experience_size,
                                                                                              setting_strategy,
                                                                                              image_height,
                                                                                              image_width))

        end_train_dataset = _load_dataset(
            dataset_folder,
            end_train_dataset_file, lambda json_dataset: _compute_end_dataset(
                json_dataset, setting_strategy, image_height, image_width))

        end_valid_dataset = _load_dataset(
            dataset_folder,
            end_valid_dataset_file, lambda json_dataset: _compute_end_dataset(
                json_dataset, setting_strategy, image_height, image_width))

        test_dataset = _load_dataset(
            dataset_folder,
            test_dataset_file, lambda json_dataset: _compute_test_dataset(
                json_dataset, setting_strategy, image_height, image_width))

    if not dataset_params["load_dataset"] \
        or (dynamic_train_dataset is None and continual) \
        or (end_train_dataset is None and not continual) \
        or (end_valid_dataset is None and not continual and use_validation):
        dynamic_train_dataset, \
        nodes_moved_in_each_experience, \
        end_train_dataset, \
        end_valid_dataset, \
        test_dataset = _create_dataset(dataset_params["dataset_folder"],
                                       image_folders,
                                       train_paths,
                                       train_poses_files,
                                       train_nodes_files,
                                       train_links_files,
                                       clustering_params,
                                       image_height,
                                       image_width,
                                       end_train_dataset_file,
                                       dynamic_train_dataset_file,
                                       experience_size,
                                       use_validation,
                                       seed,
                                       validation_params["balanced_validation"],
                                       validation_params["validation_size"],
                                       end_valid_dataset_file,
                                       test_paths,
                                       test_poses_files,
                                       test_nodes_files,
                                       test_links_files,
                                       test_dataset_file,
                                       setting_strategy
                                    )
    return dynamic_train_dataset, \
           nodes_moved_in_each_experience, \
           end_train_dataset, \
           end_valid_dataset, \
           test_dataset


__all__ = ["merge_datasets", "get_dataset", "extract_tensor_dataset"]
