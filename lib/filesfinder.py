import os

def find_files(directory, extension):
    files_set = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if file_path not in files_set:
                    files_set.add(file_path)
        for dir in dirs:
            files_set.update(find_files(os.path.join(root, dir), extension))
    return files_set

#files_list = find_files('/home/dliang/Documents/Carmen/23_stage_MISS_WIART/02_OP2Data_Resize/', 'T2H0_128.nii')

#files_list_sorted = sorted(files_list)


#for file_path in files_list_sorted:
#    print(file_path)
