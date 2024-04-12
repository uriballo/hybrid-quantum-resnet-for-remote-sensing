import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import collections

class EuroSATLoader:
    def __init__(self, root, image_size=64, batch_size=256, test_size=0.2, random_state=42, examples=1000):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.examples = examples
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        class_counts = collections.Counter(target for _, target in dataset)
        
        if self.examples > 0 and self.examples < len(dataset):
            num_per_class = self.examples // len(class_counts)
            train_indices, val_indices = [], []
            for class_label in range(len(class_counts)):
                class_indices = [i for i, (_, target) in enumerate(dataset) if target == class_label]
                class_train_indices, class_val_indices = train_test_split(class_indices, test_size=self.test_size, train_size=num_per_class, random_state=self.random_state)
                train_indices.extend(class_train_indices)
                val_indices.extend(class_val_indices)
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, val_indices)
        else:
            train_set, val_set = train_test_split(dataset, test_size=self.test_size, random_state=self.random_state, stratify=[y for _, y in dataset.samples])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.print_class_counts(train_set, "train")
        self.print_class_counts(val_set, "test")

        return train_loader, val_loader

    def print_class_counts(self, dataset, set_type):
        class_counts = collections.Counter(target for _, target in dataset)
        print(f"Class distribution in the {set_type} set:")
        for class_label, count in sorted(class_counts.items()):
            print(f"\tClass {class_label}: {count} examples")

