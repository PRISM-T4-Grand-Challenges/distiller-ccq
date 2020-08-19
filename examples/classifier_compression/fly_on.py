##GonoTontro Bomaru Biman

import yaml
import yamlordereddictloader
import numpy as np
import os
import csv
from itertools import chain

#defining dictionary at the beginning
# quantizer = 'pact_quantizer'

base_yaml = './dumping_station/ishtashin.yaml'
 

bar = 0 ##koibar train run dilam
epoch = 0 ##total no. fo epochs
qsteps = 80

clear_dir1 = 'rm -rfv ./just/*'
clear_dir2 = 'rm -rfv ./third_cifar/*'
os.system(clear_dir1)
os.system(clear_dir2)
cp_refar = 'cp -rfv base/refar0___2019.11.20-203154 third_cifar/'
cp_pact = 'cp -rfv base/pact_track.csv just/'
os.system(cp_refar)
os.system(cp_pact)
cp_yaml = 'cp -rfv dumping_station/back_ishtashin.yaml dumping_station/ishtashin.yaml'
cp_csv = 'cp -rfv dumping_station/back_data.csv dumping_station/data.csv'
os.system(cp_yaml)
os.system(cp_csv)

lr = 0.0001
#first_run = 'python compress_classifier.py -a resnet20_cifar --lr '+ str(lr) +' -b 128 -p 50 /home/mdl/mzk591/dataset/data.cifar10 -j 2 --epochs 50 --gpus 3 --vs 0 --wd 0.0002 --compress='+base_yaml+' -o third_cifar -n refar'+str(bar)
sample_run = 'python sample.py -a resnet50 -b 512 -p 50 /home/mdl/mzk591/dataset/data.imagenet -j 32 --epochs 1 --gpus 0,1,2 --vs 0 --resume third_cifar/refar0___2019.11.20-203154/refar0_checkpoint.pth.tar --compress='+base_yaml+' -o third_cifar -n sampling'

os.system(sample_run)

epoch += 1

bitmap = [8, 6, 4, 3, 2]

def dump_yaml(OrDict,name,mode):
    yaml.dump(
            OrDict,
            open(name, mode),
            Dumper=yamlordereddictloader.Dumper,
            default_flow_style=False)

def get_layers(sched_dict,quantizer):
    temp_hold = list(sched_dict['quantizers'][quantizer]['bits_overrides'].keys())
    layers = []
    for item in temp_hold:
        if 'relu' not in item:
            layers.append(item)
    return layers
            
def yaml_rw(filename,item):
    global bitmap
    try:
        # print(filename)
        sched_dict =yaml.load(open(filename), Loader=yamlordereddictloader.Loader)
        # print(sched_dict)
        temp = list(sched_dict['quantizers'])
        quantizer = temp[0]
        layers = get_layers(sched_dict,quantizer)       
        # print(layers)
        print(item)

        a = sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts']
        print(a)         
        current_config = bitmap.index(a)
        sched_dict['quantizers'][quantizer]['bits_overrides'][item]['wts'] = int(bitmap[current_config+1])
        sched_dict['quantizers'][quantizer]['bits_overrides'][item]['acts'] = int(bitmap[current_config+1])
        if 'identity_conv' in item or 'output' in item:
                pass
        else:
            acts = item[:-4] + 'activ'
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = int(bitmap[current_config+1])
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = int(bitmap[current_config+1])

        dump_yaml(sched_dict,filename,'w') ##Output YAML file name 
    except yaml.YAMLError as exc:
        print("\nFATAL parsing error while parsing the schedule configuration file %s" % filename)
        raise
        
def get_chkpt():    
    global bar
    master_chkpt = 'refar' + str(bar)
    next_chkpt = 'refar' + str(bar+1)
    return master_chkpt,next_chkpt

def get_master_path(master):
    dir_name = []
    
    for (dirpath, dirnames, filenames) in os.walk('./third_cifar'):
        dir_name.extend(dirnames)
        break
    
    for name in dir_name:
        #print(name)
        if master in name:
            print('Found it!!')
            path = 'third_cifar/'+name+'/'+master+'_checkpoint.pth.tar'
            break        
    return path
        

def get_layer_name():
    with open("./dumping_station/data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            pass
        return line[0]
    
while bar<qsteps:
    print('Hola!\n')
    ##first read the layer name
    mod_layer = get_layer_name()
    ##Read the curr_yaml
    yaml_rw(base_yaml,mod_layer)
    ##retrain for 2 epochs
    master,next=get_chkpt()
    master_path = get_master_path(master)
    print(master_path)
    
    run = 'python online_compression.py -a resnet50 --lr '+str(lr)+' -b 512 -p 100 /home/mdl/mzk591/dataset/data.imagenet -j 32 --epochs 3 --qsteps '+str(qsteps)+' --gpus 0,1,2 --vs 0 --resume '+ master_path +' --compress='+ base_yaml +' -o third_cifar -n refar'+str(bar+1)

    os.system(run)
    epoch += 3
    
    bar += 1

master,next=get_chkpt()
master_path = get_master_path(master)
print(master_path)


# lr = lr*0.82
# tune = 'python online_compression.py -a resnet18 --lr '+str(lr)+' -b 512 -p 50 /home/mdl/mzk591/dataset/data.imagenet -j 8 --epochs 40 --qsteps '+str(qsteps)+' --gpus 1 --vs 0 --resume '+ master_path +' --compress='+ base_yaml +' -o third_cifar -n refar'+str(bar+1)
# os.system(tune)
