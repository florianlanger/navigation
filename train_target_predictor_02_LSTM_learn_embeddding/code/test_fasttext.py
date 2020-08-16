import fasttext
import fasttext.util
import numpy as np
from numpy import dot
from numpy.linalg import norm



def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


fasttext.util.download_model('en', if_exists='ignore')

ft_model = fasttext.load_model('cc.en.300.bin')
print('loaded_model')
s = [
'To the right of the cube further up and behind it',
'Above the cube but slightly further back and a bit to the right.',
'To the right of the Cube slightly further back on the same height as the bottom.',
'Underneath the backside of the cube.',
'Right of the cube it close to the back.',
'To the left of the cube near the back but further down.',
'To the left of the cube in the same height as the bottom.']

embeds = np.zeros((7,300))
for i,sentence in enumerate(s):
    print(i)
    embeds[i] = ft_model.get_sentence_vector(sentence)

similarity = np.zeros((7,7))

for i in range(7):
    for j in range(7):
        print(j)
        similarity[i,j] = cos_sim(embeds[i],embeds[j])
    
print(similarity)