from torch.utils import data
import json
import torch



class Target_Predictor_Dataset(data.Dataset):
    def __init__(self, file_path,number_examples):
        self.number_examples = number_examples
        self.target_poses = torch.empty((number_examples,3)).cuda()
        self.cubes = torch.empty((number_examples,6)).cuda()
        self.descriptions_strings = []
        self.descriptions_lengths = torch.zeros(number_examples).cuda()
        with open(file_path, 'r') as csv_file:
            for i in range(number_examples):
                    line = csv_file.readline()
                    dictionary = json.loads(line)
                    self.target_poses[i] = torch.tensor(dictionary["target_pose"])
                    self.cubes[i] = torch.tensor(dictionary["cube"])
                    self.descriptions_strings.append(dictionary["description"])
                    self.descriptions_lengths[i] = len(dictionary["description"].split())
       
        # create vocab, i.e. mapping from words to integers
        self.vocab = ['<pad>'] + sorted(set([word for sentence in self.descriptions_strings for word in sentence.split()]))
        self.len_vocab = len(self.vocab)
        # create arrays of word indices of fixed length paddded with 0's in the back
        self.vectorized_descriptions = torch.zeros((number_examples,40),dtype=torch.long).cuda()
        for i in range(number_examples):
            for j,word in enumerate(self.descriptions_strings[i].split()):
                self.vectorized_descriptions[i,j] = self.vocab.index(word)

    def __len__(self):
        return self.number_examples

    def __getitem__(self,index):
        return self.cubes[index],self.vectorized_descriptions[index],self.descriptions_lengths[index],self.descriptions_strings[index],self.target_poses[index]

# training_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data/data.csv',80)

# train_loader = data.DataLoader(training_data, batch_size = 3)
# for data in train_loader:
#        print(data)
     