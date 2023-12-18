import os
import numpy as np
import matplotlib.pyplot as plt

result_dir = "./results/numpy"
lst_data = os.listdir(result_dir)

# create data list
lst_input = [f for f in lst_data if f.startswith('input')]
lst_label =  [f for f in lst_data if f.startswith('label')]
lst_output = [f for f in lst_data if f.startswith('output')]

# sorting
lst_input.sort()
lst_label.sort()
lst_output.sort()

# select sample data
id = 0
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

# show image
plt.subplot(131)
plt.imshow(label)
plt.title('label')

plt.subplot(132)
plt.imshow(input)
plt.title('input')

plt.subplot(133)
plt.imshow(output)
plt.title('output')

plt.show()