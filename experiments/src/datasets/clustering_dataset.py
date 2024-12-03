from clustering import get_clustering
from .dataset_utils import _load_data_for_clustering
from .path_dataset import PathDataset
import globals

class ClusteringDataset:
    def __init__(self, params):
        
        dataset_params = params["dataset"]
        
        train_sequences = params["dataset"]["train"]["sequences"]
        test_sequences = params["dataset"]["test"]["sequences"]

        dataset_params = params["dataset"]
        self.image_width = dataset_params["image_width"]
        self.image_height = dataset_params["image_height"]
        
        
        train_params = dataset_params["train"]
        test_params = dataset_params["test"]

        clustering_params = params["clustering"]
        dataset_type = clustering_params["dataset_type"]
        
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

        self.experience_size = train_params["experience_size"]
        
        clustering_params = params["clustering"]
        self.clustering = get_clustering(params["clustering"]["type"], **clustering_params)  
        self.train_nodes, self.train_links = _load_data_for_clustering(
            train_sequences, image_folders, train_poses_files, dataset_type,
            train_nodes_files, train_links_files)
        
        self.test_nodes, self.test_links = _load_data_for_clustering(
            test_sequences, image_folders, test_poses_files, dataset_type,
            test_nodes_files, test_links_files)
        
        self.exp = 0
        self.num_classes = 0
        self.current_sequence = 0
        self.new_sequence = True
        self.n_sequences = len(train_sequences)
        self.end = False
    
    def get_next_experience(self):
        if self.end:
            return None, None, None
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Get next experience\n")
        moved = {}
        experience = []
        test = []
        if self.new_sequence:
            self.clustering.set_current_sequence(self.train_nodes[self.current_sequence], self.train_links[self.current_sequence] if self.train_links is not None else None)
            self.new_sequence = False
        cont = self.clustering.clusterize_next_experience(self.experience_size)
        if not cont:
            self.new_sequence = True
            self.current_sequence += 1
            if self.current_sequence > self.n_sequences:
                self.end = True
        moved.update(self.clustering.experience_moved)
        ids = []
        images = []
        labels = []
        for n in self.clustering.nodes[self.clustering.current_node_idx - self.experience_size:self.clustering.current_node_idx]:
            ids.append(int(n.id))
            images.append(n.images)
            labels.append(int(n.region_id))   
        experience = PathDataset(ids, images, labels, self.image_height, self.image_width)
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Ended clustering of experience {self.exp} of sequence {self.current_sequence} with exp size = {len(experience)}\n")
        for j in range(len(self.test_nodes)):
            self.clustering.align_sequence_until_now(self.test_nodes[j])
            ids = []
            images = []
            labels = []
            for n in self.test_nodes[j]:
                if n.region_id != -1:
                    ids.append(int(n.id))
                    images.append(n.images)
                    labels.append(int(n.region_id)) 
                if n.region_id >= self.num_classes:
                    self.num_classes += 1
            test.append(PathDataset(ids, images, labels, self.image_height, self.image_width))
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"Ended alignment of sequence {j}\n")
        self.exp += 1
        return experience, moved, test
    
__all__ = ["ClusteringDataset"]