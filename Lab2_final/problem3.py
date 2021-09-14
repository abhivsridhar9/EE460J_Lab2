import glob
import os

import pandas as pd

# part a
year=input('Year: ')
k=input('Top #: ')
df=pd.read_csv('../Names/yob'+year+'.txt',index_col=0,header=None)

print(df.head(int(k)))

# part b
folder_path = '../Names'
input_name = input('Name: ')
frequency = 0
freq_name = None
for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    df = pd.read_csv(filename, index_col=None, header=None)
    for name in df.itertuples():
        #print(name[1])
        if (name[1] == input_name):
            frequency+=name[3]
            #print(name[1], name[2])
            freq_name = name[1]
print("Frequency of",freq_name, frequency)


# part c


#folder_path = '../Names'
input_name = input('Name: ')
freq_year = input("Year:")
#input_name = name
#freq_year = year
fem_frequency = 0
male_frequency = 0
relative_frequency = 0
count = 0
filename = "../Names/yob"+str(freq_year)+".txt"
df = pd.read_csv(filename, index_col=None, header=None)
for name in df.itertuples():
    count = count + name[3]
    if(name[1] == input_name):
        if(name[2] == 'M'):
            male_frequency = male_frequency + name[3]
        else:
            fem_frequency = fem_frequency + name[3]
print("Name:", input_name)
print("Male Frequency:", male_frequency/count)
print("Female Frequency:", fem_frequency/count)



# part d
folder_path = '../Names'
#input_name = input('Name: ')
male_frequency = 0
fem_frequency = 0
freq_name = None
frequency_dict = {}
for x in range(1880,2015):
    df = pd.read_csv("../Names/yob"+str(x)+".txt", index_col=None, header=None)
    male_frequency = 0
    fem_frequency = 0
    rel_frequency = 0
    year = x
    for name in df.itertuples():
        if (name[2] == 'M'):
            try:
                if (frequency_dict[name[1]]):
                        temp_list = frequency_dict[name[1]]
                        length = len(temp_list)
                        if(temp_list[length - 1][1] == 0 and temp_list[length - 1][2] != 0):
                            temp_list[length - 1][1] = name[3]
                            frequency_dict[name[1]] = temp_list
                        else:
                            temp_list = frequency_dict[name[1]]
                            tuple = [x, name[3], 0]
                            temp_list.append(tuple)
                            frequency_dict[name[1]] = temp_list
            except KeyError:
                list = []
                tuple = [x, name[3], 0]
                list.append(tuple)
                frequency_dict[name[1]] = list
        else:
            if (name[2] == 'F'):
                try:
                    if (frequency_dict[name[1]]):
                        temp_list = frequency_dict[name[1]]
                        length = len(temp_list)

                        if(temp_list[length-1][2] == 0 and temp_list[length-1][1] != 0):
                            temp_list[length - 1][2] = name[3]
                            frequency_dict[name[1]] = temp_list
                        else:
                            temp_list = frequency_dict[name[1]]
                            tuple = [x, 0, name[3]]
                            temp_list.append(tuple)
                            frequency_dict[name[1]] = temp_list
                except KeyError:
                    list = []
                    tuple = [x, 0, name[3]]
                    list.append(tuple)
                    frequency_dict[name[1]] = list
final_list = []
for name in frequency_dict.keys():
    name_list = frequency_dict[name]
    initial_sign_num = frequency_dict[name][0][1] - frequency_dict[name][0][2]
    if(initial_sign_num > 0):
        initial_sign = "positive"
    else:
        initial_sign = "negative"
    for year in name_list:
        sign = ""
        relative_frequency = year[1] - year[2]
        if(relative_frequency > 0):
            sign = "positive"
        else:
            sign = "negative"
        if(initial_sign != sign):

            final_list.append(name)
            break

print(final_list)



