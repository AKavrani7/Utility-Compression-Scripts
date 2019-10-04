import scipy.io as sio
import numpy as np

from src.load_my_data import load_mat_file
from src.RLS_filter import generate_local_difference, get_residual_error_by_band, get_residual_error

############## 1. Read Dataset ######################################
print("1.Read Dataset- Create Data_Array\nShape:")

corrected_dataset = 'Dataset/Indian_pines/Indian_pines_corrected.mat'
dataset = 'Dataset/Indian_pines/Indian_pines.mat'
gt_dataset = 'Dataset/Indian_pines/Indian_pines_gt.mat'

corrected_data = load_mat_file(corrected_dataset)
#print(corrected_data)
corrected_data_array = corrected_data['indian_pines_corrected']
print(corrected_data_array.shape)

'''
data = load_mat_file(dataset)
#print(data)
data_array = data['indian_pines']
print(data_array.shape)

gt_data = load_mat_file(gt_dataset)
#print(gt_data)
gt_data_array = gt_data['indian_pines_gt']
print(gt_data_array.shape)

'''
print("\nDone 1.Read Dataset- Created Data_Array\n")
#################### *************** ##############

################# 2. Local Difference Matrix ########
print("2. Create local difference matrix")

local_diff_matrix = []
no_of_bands = 10

print("no of bands: " + str(no_of_bands))

for i in range(no_of_bands):
    print("\n\tband no: " + str(i))
    local_difference = generate_local_difference(corrected_data_array[:,:,i], 1)
    print("\tlen of local diff " + str(len(local_difference)) )

    local_diff_matrix.append(local_difference)
    print("\tlen of local diff matrix " + str(len(local_diff_matrix)) )


local_diff_matrix = np.array(local_diff_matrix)
print("local_diff_matrix generated")
shape = local_diff_matrix.shape
local_diff_matrix = np.reshape( local_diff_matrix, (shape[1],shape[0]))
shape = local_diff_matrix.shape
print("Shape of local difference matrix " + str(shape))
print("2. Done Created local Difference Matrix\n")


################### Step 3. Residual Error ################
print("3. Find Residual Error Matrix\n")

local_diff_matrix = local_diff_matrix[0:20, :]
print("Considering shape of local_diff_matrix: " + str(local_diff_matrix.shape))
ez_matrix = get_residual_error( local_diff_matrix, 4)
shape = ez_matrix.shape
print("Shape of ez_matrix or residual error matrix" + str(shape))
print("\n 3. Got the Residual error matrix")

################# ************************ ##################