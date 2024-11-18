def get_datasets(dataset_name, data_root, train, transforms=None):
    if dataset_name == 'DRIVE':
        from ..DRIVE_dataset import DRIVE_dataset
        dataset = DRIVE_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
    elif dataset_name == 'ICCAD':
        from ..ICCAD_dataset import CustomDataset
        
        dataset = CustomDataset(csv_dir=data_root,train=train)

        num_return = 0 # ??

    
    else:
        raise NotImplementedError

    return dataset, num_return
