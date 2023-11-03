
class StaticDataset(Dataset):
    def __init__(self, root_path, flag='train', static_data_path='static_data.csv'):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.static_data_path = static_data_path
        self.__read_static_data__()

    def __read_static_data__(self):
        df_static = pd.read_csv(os.path.join(self.root_path, self.static_data_path))
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.static_data = df_static.values[border1:border2]

    def __getitem__(self, index):
        return self.static_data[index]

    def __len__(self):
        return len(self.static_data)
