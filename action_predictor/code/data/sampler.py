import torch
import networkx as nx


class Sampler(torch.utils.data.Sampler):
      def __init__(self,graph_path,config):
            self.config = config
            self.pairs_per_epoch = config["training"]["pairs_per_epoch"]
            self.ratio_hard_pairs = config["sampler"]["ratio_hard_pairs"]
            self.number_hard_pairs = int(self.pairs_per_epoch * self.ratio_hard_pairs)
            self.ratio_terminate_pairs = config["sampler"]["ratio_terminate_pairs"]
            # Note ratio terminate pairs is defined w.r.t. whole data set 
            self.min_new_terminate_pairs = int(self.ratio_terminate_pairs * self.pairs_per_epoch)
            self.graph = nx.read_gpickle(graph_path)
            self.number_non_terminate_nodes = int(self.graph.number_of_nodes() / 2)
            self.random()

      def update_sampler(self,hard_pairs):
            hard_pairs = hard_pairs[:self.number_hard_pairs].long()
            random_pairs = self.sample_allowed_pairs(int(self.pairs_per_epoch-self.number_hard_pairs))
            if self.min_new_terminate_pairs != 0:
                  random_pairs[:self.min_new_terminate_pairs] = self.sample_allowed_pairs(self.min_new_terminate_pairs,terminate=True)
            pairs = torch.cat((random_pairs,hard_pairs))
            self.pairs = pairs[torch.randperm(self.pairs_per_epoch)]

      def random(self):
            if self.min_new_terminate_pairs == 0:
                  self.pairs = self.sample_allowed_pairs(self.pairs_per_epoch)
            else:
                  self.pairs = torch.empty((self.pairs_per_epoch,2),dtype=torch.int64).cuda()
                  self.pairs[:self.min_new_terminate_pairs] = self.sample_allowed_pairs(self.min_new_terminate_pairs,terminate=True)
                  self.pairs[self.min_new_terminate_pairs:] = self.sample_allowed_pairs(self.pairs_per_epoch - self.min_new_terminate_pairs)
                  self.pairs = self.pairs[torch.randperm(self.pairs_per_epoch)]
                  
      def sample_allowed_pairs(self,number, terminate = False):
            pairs = torch.empty(number,2,dtype=torch.int64).cuda()
            for i in range(number):
                  if terminate == False:
                        pairs[i,0] = self.sample_index_outside_no_fly()
                        pairs[i,1] = self.sample_index_outside_no_fly() + self.number_non_terminate_nodes
                  if terminate == True:
                        pairs[i,0] = self.sample_index_outside_no_fly()
                        pairs[i,1] = pairs[i,0] + self.number_non_terminate_nodes
            return pairs

      def sample_index_outside_no_fly(self):
            while True:
                  index = torch.randint(self.number_non_terminate_nodes,(1,)).cuda()
                  if self.graph.nodes[index.item()]["flyable"]:
                        break
            return index

      
      def __iter__(self):
          self.counter = -1
          return self

      def __len__(self):
        return self.pairs_per_epoch

      def __next__(self):            
            if self.counter < self.pairs_per_epoch-1:
                  self.counter += 1
                  return self.pairs[self.counter]
                  
            else:
                  raise StopIteration 
