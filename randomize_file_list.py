from random import shuffle

with open('/home/student/Data_2_1/file_list.txt') as f:
    h5_files = f.readlines()

h5_files = [x.strip() for x in h5_files]
shuffle(h5_files)
num_train = int(len(h5_files) * 0.8)
train_files = h5_files[:num_train]
test_files = h5_files[num_train:]

with open('train_files.txt', 'w') as f:
    for item in train_files:
        f.write("%s\n" % item)

with open('test_files.txt', 'w') as f:
    for item in test_files:
        f.write("%s\n" % item)