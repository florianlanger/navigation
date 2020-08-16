import gensim.downloader as api
import os 
import json
import pickle
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

cached_word_to_vector = {'dgfhf':np.array([3,4,5])}


with open('{}/cached_word_to_vector.json'.format(dir_path), 'wb') as fp:
                pickle.dump(cached_word_to_vector, fp)

wv = api.load("glove-twitter-25")
with open('{}/../target_pose/training_data/data.csv'.format(dir_path), 'r') as csv_file:
        for i in range(89):
            line = csv_file.readline()
            train_dict = json.loads(line)
            words = train_dict["description"].split()
            for word in words:
                if word not in cached_word_to_vector:
                    try:
                        cached_word_to_vector[word] = wv[word]
                    except KeyError:
                        print(word)

            with open('{}/cached_word_to_vector.json'.format(dir_path), 'wb') as fp:
                pickle.dump(cached_word_to_vector, fp)
            print(i)


