import glob
import os

import pandas as pd

# # part a
# year=input('Year: ')
# k=input('Top #: ')
# df=pd.read_csv('../Names/yob'+year+'.txt',index_col=0,header=None)
#
# print(df.head(int(k)))


# # part b
# folder_path = '../Names'
# input_name = input('Name: ')
# for filename in glob.glob(os.path.join(folder_path, '*.txt')):
#     df = pd.read_csv(filename, index_col=None, header=None)
#     for name in df.iterrows():
#         if (name[1][0] == input_name):
#             print(frequency)
#
#

# # part c
# folder_path = '../Names'
# input_name = input('Name: ')
# frequency = 0
# year_total = 0
# for filename in glob.glob(os.path.join(folder_path, '*.txt')):
#     df = pd.read_csv(filename, index_col=None, header=None)
#     for name in df.iterrows():
#         year_total += name[1][2]
#     for name in df.iterrows():
#         if (name[1][0] == input_name):
#             frequency = name[1][2] / year_total
#     print(frequency)
#     frequency = 0
#     year_total = 0

# part d
df=pd.read_csv('../Names/yob2004.txt')
male_list=[]
female_list=[]
year_total=0
for item in df.iterrows():
    if(item[1][1]=='M'):
        male_list.append(item[1][0])
    elif(item[1][1]=='F'):
        female_list.append(item[1][0])
    year_total+=item[1][2]


intersection_set = set.intersection(set(male_list), set(female_list))
intersection_list = list(intersection_set)
combined_list=[]
# for item in df.iterrows():
#     temp_list=[]
#     for i in intersection_list:
#         if(i==item[1][0]):
#             if(item[1][1]=='M'):
#                 temp_list.append(item[1][0])
#                 temp_list.append(item[1][2]/year_total)
#     for i in intersection_list:
#         if(i==item[1][0]):
#             if(item[1][1]=='F'):
#                 temp_list.append(item[1][2]/year_total)
#     combined_list.append(temp_list)

print(df.values)
print((combined_list))
dict.fromkeys([1,2,3])
