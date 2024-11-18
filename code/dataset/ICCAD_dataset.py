import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__ (self, csv_dir, train):
        self.csv_dir = csv_dir

        self.idx_map = {}
        self.train = train

        # self.ir_csv = [f for f in os.listdir(csv_dir) if f.endswith("_ir_drop.csv")]
        # self.distance_csv = [f for f in os.listdir(csv_dir) if f.endswith("_eff_dist.csv")]
        # self.density_csv = [f for f in os.listdir(csv_dir) if f.endswith("_pdn_density.csv")]
        # self.current_csv = [f for f in os.listdir(csv_dir) if f.endswith("_current.csv")]

        if self.train:
            ir_csv = [f for f in os.listdir(csv_dir) if f.endswith("_ir_drop.csv")]

            for file_name in ir_csv:
                num_part = file_name[len('current_map'):-len('_ir_drop.csv')]
                try:
                    idx = int(num_part)  
                    self.idx_map[idx] = num_part
                except ValueError:
                    pass  
        else:
            cnt = 0

            test_case = [f for f in os.listdir(csv_dir) if f.startswith("testcase")]
            
            for file in test_case:
                num_part = file[len("testcase"):]
                try:
                    self.idx_map[cnt] = num_part
                    cnt += 1
                except ValueError:
                    pass 

        self.mean_list = [6.248153293943716e-08, 14.041562023946096, 1.4920440930657963]
        self.mean_ir = 0.001139045091611615
        self.std_list = [5.5081556620700834e-08, 5.714110412062309, 1.1141596211756297]
        self.std_ir = 0.000507999665482062

        self.feature_transform = transforms.Normalize(self.mean_list,self.std_list)
        self.ir_transform = transforms.Normalize(self.mean_ir,self.std_ir)

        # ## compute mean and std
        # self.mean_ir = 0
        # for ir_csv in self.ir_csv:
        #     data = pd.read_csv(os.path.join(self.csv_dir,ir_csv)).values
        #     data = data.flatten()
        #     self.mean_ir += np.mean(data)

        # self.mean_ir /= len(self.ir_csv)
        
        # self.mean_list = []

        # mean = 0

        # input_csv = [self.current_csv,self.distance_csv,self.density_csv]

        # for i in range(3):
        #     mean = 0
        #     for input in input_csv[i]:
        #         data = pd.read_csv(os.path.join(self.csv_dir,input)).values
        #         data = data.flatten()
        #         mean += np.mean(data)    

        #     self.mean_list.append(mean/len(self.current_csv))         


        # ## compute std

        # self.std_ir = 0
        # self.std_list = []
        # diff_ir = 0
        # N = 0
        # for ir_csv in self.ir_csv:
        #     data = pd.read_csv(os.path.join(self.csv_dir,ir_csv)).values
        #     data = data.flatten()
            
        #     diff_ir += np.sum(np.power(data - self.mean_ir,2))      
        #     N += len(data)

        # self.std_ir = np.sqrt(diff_ir / N) 

        # for i in range(3):
        #     diff = 0
        #     for input in input_csv[i]:
        #         data = pd.read_csv(os.path.join(self.csv_dir,input)).values
        #         data = data.flatten()
                
        #         diff += np.sum(np.power(data - self.mean_list[i],2))       

        #     std = np.sqrt(diff / N)

        #     self.std_list.append(std) 


        # print(self.mean_list)
        # print(self.mean_ir)
        # print(self.std_list)
        # print(self.std_ir)

    def __len__ (self):
        return len(self.idx_map)

    def __getitem__ (self, idx):

        file_idx = self.idx_map[idx]


        ir_drop_path = ''
        distance_path = ''
        density_path = ''
        current_path = ''


        if self.train:
            ir_drop_path = os.path.join(self.csv_dir,f"current_map{file_idx}_ir_drop.csv")
            distance_path = os.path.join(self.csv_dir,f"current_map{file_idx}_eff_dist.csv")
            density_path = os.path.join(self.csv_dir,f"current_map{file_idx}_pdn_density.csv")
            current_path = os.path.join(self.csv_dir,f"current_map{file_idx}_current.csv")
        else:
            ir_drop_path = os.path.join(self.csv_dir,f"testcase{file_idx}","ir_drop_map.csv")
            distance_path = os.path.join(self.csv_dir,f"testcase{file_idx}","eff_dist_map.csv")
            density_path = os.path.join(self.csv_dir,f"testcase{file_idx}","pdn_density.csv")
            current_path = os.path.join(self.csv_dir,f"testcase{file_idx}","current_map.csv")



        ir_df = pd.read_csv(ir_drop_path,delimiter=',')
        dis_df = pd.read_csv(distance_path,delimiter=',')
        density_df = pd.read_csv(density_path,delimiter=',')
        currrent_df = pd.read_csv(current_path,delimiter=',')

        ir_tensor = torch.tensor(ir_df.to_numpy(), dtype = torch.float32 )
        dis_tensor = torch.tensor(dis_df.to_numpy(), dtype = torch.float32 )
        density_tensor = torch.tensor(density_df.to_numpy(),dtype = torch.float32 )
        current_tensor = torch.tensor(currrent_df.to_numpy(),dtype = torch.float32 )

        feature_tensor = torch.stack([current_tensor,dis_tensor,density_tensor],dim=0)


        ir_tensor = torch.reshape(ir_tensor,(1,ir_tensor.shape[0],ir_tensor.shape[1]))


        return self.feature_transform(feature_tensor),  self.ir_transform(ir_tensor)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = CustomDataset('./dataset/fake-circuit-data_20230623/fake-circuit-data',True)
    test_dataset = CustomDataset('./dataset/real-circuit-data_20230615',False)


    train_dl = DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_dl = DataLoader(test_dataset,batch_size=1)

    print(len(test_dataset))

    f, l = next(iter(train_dl))

    print(f.shape)
    print(l.shape)

    

