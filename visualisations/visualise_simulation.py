
import numpy as np
from matplotlib import pyplot as plt
import statistics

predicted_differences = np.array([[0.7,0.5385,0.5385,0.5385,0.4123],
                                   [1.64,0.8602,0.6782,0.6782,0.5099],
                                    [1.257,1.27,1.43,1.43,0.96],
                                    [0.36,0.3,0.3,0.41,0.22],
                                    [1.16,1.16,1.28,1.28,0.447],
                                    [0.83,0.83,0.36,0.41,0.41]
                                    ])

actual_differences = np.array([[0.6708,0.6457,0.522,0.522,0.39],
                                [1.705,0.3742,0.5477,0.7265,0.81],
                                [1.26,1.26,1.36,1.53,1.18],
                                [0.37,0.25,0.25,0.48,0.28],
                                [1.07,1.17,1.45,1.45,0.707],
                                [0.6325,0.632,0.4594,0.48,0.48]
])

mean_predicted = np.mean(predicted_differences,axis=0)
median_predicted = np.empty(5)
for i in range(5):
    median_predicted[i] = statistics.median(predicted_differences[:,i])

mean_actual = np.mean(predicted_differences,axis=0)
median_actual = np.empty(5)
for i in range(5):
    median_actual[i] = statistics.median(predicted_differences[:,i])

print(mean_actual)


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# x = [1,2,3,4,5]
# for i in range(6):
#     ax.plot(x,predicted_differences[i],label="Scenario {}".format(i+1),linestyle='dashed')

# ax.plot(x,mean_predicted,label='mean',color='blue')
# ax.plot(x,median_predicted,label='median',color='black')

# ax.legend()

# ax.set_xlabel('Number of descriptions provided')
# ax.set_ylabel('Distance to target (m)')
# ax.set_xticks([1,2,3,4,5])

# fig.savefig('simulation_visualisation/predicted_differences.png',dpi=400,bbox_inches = "tight")
# plt.close(fig)

