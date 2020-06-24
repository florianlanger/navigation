import os
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, 'Name of the experiment folder')


def plot(history_dict,exp_path):
    for counter in range(history_dict["counter"]):
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(mpimg.imread(exp_path + '/images/'+ history_dict["image_names"][counter]))
        fig.add_subplot(1,2,2)
        plt.imshow(mpimg.imread(exp_path + '/renders/' + history_dict["image_names"][counter]))
        plt.figtext(0.1, 0.95, 'Predicted Pose: {}'.format(list(history_dict["predicted_poses"][counter])), wrap=True, fontsize=12)
        fig.savefig(exp_path + '/combined/' + history_dict["image_names"][counter])
        plt.close(fig)


def main(argv):
    del argv
    dir_path = os.path.dirname(os.path.realpath(__file__))

    exp_path = dir_path + '/../experiments/' + FLAGS.name
    #os.mkdir(exp_path + '/combined')

    history_dict = np.load(exp_path + '/history_dict.npy',allow_pickle=True).item()

    plot(history_dict,exp_path)


    
   
    
    
if __name__ == "__main__":
    app.run(main)