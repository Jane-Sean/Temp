# import os
# rootpath = '../datasets/Dcase2020/fan'
# for root, dirs, files in os.walk(rootpath, topdown=False):
#     print(f'root:{root},dirs:{dirs},files:{files}', sep="??")
    # for name in files:
    #     print(os.path.join(root, name),sep="|||")
    # for name in dirs:
    #     print(os.path.join(root, name))

# import glob
# file_path_pattern = "./*.*"
# list1 = glob.glob(file_path_pattern)
# print(list1)

import torch
print(torch.version.cuda)
