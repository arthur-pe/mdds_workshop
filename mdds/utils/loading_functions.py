from datetime import datetime
import os
import yaml
import shutil


def make_directory(directory_name='runs', sub_directory=None):
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
    directory = './'+directory_name+'/all/' + date_time
    if not os.path.exists(directory):
        os.makedirs(directory)

    if sub_directory is not None:
        for i in sub_directory:
            if not os.path.exists(directory+'/'+i):
                os.makedirs(directory+'/'+i)

    print('directory:', directory)

    return directory


def load_yaml(load_directory, directory, parameter_file='/parameters.yaml'):
    #Parameters loading
    with open(load_directory+parameter_file, 'r') as f:
        parameters = yaml.safe_load(f)
    for i in parameters:
        print(i, ':', parameters[i])
    shutil.copyfile(load_directory+parameter_file, directory+parameter_file)

    return parameters