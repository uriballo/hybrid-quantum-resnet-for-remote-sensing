import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import collections
import numpy as np

class EuroSATLoader:
    def __init__(self, root, num_classes=None, image_size=64, batch_size=256, test_size=0.2, random_state=42, examples=1000):
        self.root = root
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.examples = examples
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        original_samples = dataset.samples
        
        if self.num_classes is not None:
            class_counts = collections.Counter(target for _, target in original_samples)
            allowed_classes = list(class_counts.keys())[:self.num_classes]
            filtered_indices = [i for i, (_, target) in enumerate(original_samples) if target in allowed_classes]
            dataset = Subset(dataset, filtered_indices)

        # Handling subsets when limiting examples per class
        if self.examples > 0:
            total_examples = min(self.examples, len(dataset))
            per_class_limit = total_examples // len(np.unique([dataset.dataset.targets[idx] for idx in dataset.indices]))
            train_indices, val_indices = [], []
            for class_label in np.unique([dataset.dataset.targets[idx] for idx in dataset.indices]):
                class_indices = [idx for idx in dataset.indices if dataset.dataset.targets[idx] == class_label]
                if len(class_indices) > per_class_limit:
                    class_indices = np.random.choice(class_indices, per_class_limit, replace=False)
                if len(class_indices) >= 2:  # Ensure enough samples for split
                    class_train_indices, class_val_indices = train_test_split(class_indices, test_size=self.test_size, random_state=self.random_state)
                    train_indices.extend(class_train_indices)
                    val_indices.extend(class_val_indices)
            train_set = Subset(dataset.dataset, train_indices)
            val_set = Subset(dataset.dataset, val_indices)
        else:
            # Directly split all indices if no example limit is set
            targets = [dataset.dataset.targets[idx] for idx in dataset.indices]
            train_indices, val_indices = train_test_split(dataset.indices, test_size=self.test_size, random_state=self.random_state, stratify=targets)
            train_set = Subset(dataset.dataset, train_indices)
            val_set = Subset(dataset.dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.print_class_counts(train_set, "train")
        self.print_class_counts(val_set, "test")

        return train_loader, val_loader

    def print_class_counts(self, dataset, set_type):
        class_counts = collections.Counter(dataset.dataset.targets[idx] for idx in dataset.indices)
        print(f"Class distribution in the {set_type} set:")
        for class_label, count in sorted(class_counts.items()):
            print(f"\tClass {class_label}: {count} examples")


    def print_class_counts(self, dataset, set_type):
        class_counts = collections.Counter(dataset.dataset.targets[idx] for idx in dataset.indices)
        print(f"Class distribution in the {set_type} set:")
        for class_label, count in sorted(class_counts.items()):
            print(f"\tClass {class_label}: {count} examples")
