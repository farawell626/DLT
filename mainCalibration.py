import numpy as np
import calibmarker
from Calibration import DLTcalib
from Reconstruction import DLTrecon
import pickle
from ThreeDtest import DrawXTPYTP, SequenceFitting, ThreeDApproximate,Drawline,ReadTxt

# List of image paths
image_paths = ['./testData/test02.mp4']

xyz = [[0, -9, 0], [0, 9, 0], [9, 9, 0], [9, -9, 0], 
       [0,0, 2.43],[9,0, 2.43]
       #[6, 0, 0], [12, 0, 18], [9, 0, 18], [9, 0, 0]
       ]
'''
[[0, 0, 0], [18, 0, 0], [18, 0, 9], [0, 0, 9],[9, 2.43, 0],[9, 2.43, 9]
'''
nd = len(xyz[0])

# Call the returnUV function from calibmarker module to get marker coordinates
#uv = calibmarker.returnUV(image_paths,0)
#uv = [[(291, 280), (94, 706), (1188, 705), (989, 279), (212, 218), (1072, 221)]]
uv = [[(455, 437), (162, 1060), (1776, 1058), (1484, 440), (338, 345), (1609, 344)]] #02
print("UV: \n")
print(uv)
print("\n")

nc = len(uv)

calib_matrices = []  # List to store calibration matrices

for i in range(len(uv)):
    Li, err = DLTcalib(nd, xyz, uv[i])
    calib_matrices.append(Li)

Ls = []
for matrix in calib_matrices:
    Ls.append(matrix.tolist())  # Convert NumPy array to list
    # Use Ls as needed for further processing
print("Calibration matrices:")
#print(Ls)

lsar = np.array(Ls)
H = lsar.reshape(3, 4)
print(H)
print("\n")
'''
threeD = np.array([[-2],[0],[2],[1]])   
a = np.dot(H,threeD)
print ("Input ({},{},{},1), check :\n".format(threeD[0],threeD[1],threeD[2]) + str(a)+' (not Norm)\n')
a = a/a[2]
print ("Input ({},{},{},1), check:\n".format(threeD[0],threeD[1],threeD[2]) + str(a)+'\n')

HI= np.linalg.pinv(H)
twoD = np.array([[1383],[194],[1]])   
b = np.dot(HI,twoD)
print ("Input ({},{},1), check :\n".format(twoD[0],twoD[1]) + str(b)+' (not Norm)\n')
b = b/b[3]
print ("Input ({},{},1), check:\n".format(twoD[0],twoD[1]) + str(b)+'\n')
'''
#print(round(1/30.0,3))

path = "./testData/test02.txt"
#DrawXTPYTP(H,path)
ThreeDApproximate(H,path)
print("\n Fitting: \n")
SequenceFitting(H,path)

# Save Ls to a pickle file
# with open("calibration_params.pkl", "wb") as f:
#     pickle.dump(Ls, f)

# if nc != len(uv):
#     raise ValueError("Invalid number of cameras.")

# XYZ = np.zeros((len(xyz), nd))
# for i in range(len(uv[0])):
#     XYZ[i, :] = DLTrecon(nd, nc, Ls, [uv[j][i] for j in range(nc)])
# print('Reconstruction of the same %d points based on %d views and the camera calibration parameters:' % (len(xyz), nc))
# print(XYZ)
# print('Mean error of the point reconstruction using the DLT (error in cm):')
# print(np.mean(np.sqrt(np.sum((np.array(XYZ) - np.array(xyz))**2, 1))))
