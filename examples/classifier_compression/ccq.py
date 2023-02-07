'''
Wrapper script over intellabs distiller 
implementing the CCQ
'''
import argparse
import yaml
import yamlordereddictloader
import numpy as np
import os
import csv
from itertools import chain

#base directory for pretrained models to load from
base_dir = "./base"

#yaml dir
yaml_dir = "./yaml_dir"

# logdir name
logdir = "./logdir"

# to save intermediate yaml schedules, track training and learning parameters
dump_dir = "./dump_space"

def cp_files(opt)->None:
    clear_dir1 = f'rm -rfv {dump_dir}/*'
    clear_dir2 = f'rm -rfv {logdir}/*'
    os.system(clear_dir1)
    os.system(clear_dir2)
    cp_ckpt = f'cp -rfv {base_dir}/{opt.name}* {logdir}/'
    cp_pact = f'cp -rfv {base_dir}/pact_track.csv {dump_dir}/'
    os.system(cp_ckpt)
    os.system(cp_pact)
    cp_yaml = f'cp -rfv {yaml_dir}/base_q_schedule.yaml {dump_dir}/{opt.base_yaml}'
    touch_csv = f'touch {dump_dir}/layer_to_quantize.csv'
    mkdir_yaml_save = f'mkdir {dump_dir}/yaml_save'
    os.system(cp_yaml)
    os.system(touch_csv)
    os.system(mkdir_yaml_save)


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
            
def yaml_rw(yaml_schedule, item, current_qstep):
    
    bitmap = [8, 6, 4, 3, 2]
    
    try:
        # print(filename)
        sched_dict =yaml.load(open(yaml_schedule), Loader=yamlordereddictloader.Loader)
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
        if(item == 'conv1'):
            acts = 'relu'
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = int(bitmap[current_config+1])
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = int(bitmap[current_config+1])
        elif 'conv' in item:
            acts = item.replace('conv','relu')
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['wts'] = int(bitmap[current_config+1])
            sched_dict['quantizers'][quantizer]['bits_overrides'][acts]['acts'] = int(bitmap[current_config+1])

        dump_yaml(sched_dict,yaml_schedule,'w') ##Output YAML file name
        archive_path = os.path.join(f'{dump_dir}/yaml_save',f'yaml_schedule_{current_qstep + 1}.yaml')
        dump_yaml(sched_dict,archive_path,'w') ##Output YAML file name 
    
    except yaml.YAMLError as exc:
        print("\nFATAL parsing error while parsing the schedule configuration file %s" % yaml_schedule)
        raise

def get_ckpt_path(opt, current_qstep):
    
    ckpt_name = f'{opt.name}{current_qstep}'

    dir_name = []
    
    for (dirpath, dirnames, filenames) in os.walk(logdir):
        dir_name.extend(dirnames)
        break
    
    for name in dir_name:
        #print(name)
        if ckpt_name in name:
            print('Found it!!')
            path = os.path.join(logdir,name,f'{ckpt_name}_checkpoint.pth.tar')
            break
    return path
        

def get_layer_name():
    path = os.path.join(dump_dir,'layer_to_quantize.csv')
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            print()
        return line[0]

def getOpt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--qsteps", type=int, default=0, help="how many quantization steps to run in total")
    parser.add_argument("--mini_epochs", type=int, default=10, help="number of mini epochs for a recovery stage")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--model", type=str, default='resnet18', required = True, help="name of the model (resnet18 | resnet50)")
    parser.add_argument("--dataset_path", type=str, default='/home/mdl/mzk591/dataset/data.imagenet', required=False, help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpus", metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default='resq', help='Experiment name')
    parser.add_argument("--base_yaml", type=str, default='base_q_schedule.yaml', required=False, help="Path to base YAML compression schedule")
    
    return parser.parse_args()

def main():
    
    opt = getOpt()
    current_qstep = 0;
    
    if(current_qstep==0):
        # copy necessary files
        cp_files(opt)

        # get the checkpoint path maintaining the current qstep
        ckpt_path=get_ckpt_path(opt, current_qstep)

        yaml_path = os.path.join(dump_dir, opt.base_yaml)
        # sample if not sampled before
        sample_run = 'python sample.py -a ' + opt.model + ' -b ' + str(opt.batch_size) + ' -p 100 ' + opt.dataset_path + ' -j ' + str(opt.n_cpu) + ' --epochs 1 --gpus ' + str(opt.gpus) + \
                    ' --vs 0 --resume ' + ckpt_path + ' --compress=' + yaml_path + ' -o '+ logdir + ' -n sampling'
        os.system(sample_run)
        # print(sample_run)
    
    while current_qstep<opt.qsteps:
        # First read the layer name to modify
        layer_to_quantize = get_layer_name()
        
        # Read the current YAML quantization schedule and modify the specific layer
        yaml_path = os.path.join(dump_dir, opt.base_yaml)
        yaml_rw(yaml_path, layer_to_quantize, current_qstep)

        ckpt_path=get_ckpt_path(opt, current_qstep)
        # print(ckpt_path)
        
        yaml_path = os.path.join(dump_dir, opt.base_yaml)
        run = 'python online_compression.py -a ' + opt.model + ' --lr '+ str(opt.lr) +' -b ' + str(opt.batch_size) + ' -p 100 ' + opt.dataset_path + ' -j ' + str(opt.n_cpu) + \
                ' --epochs 3 --qsteps '+ str(opt.qsteps) +' --gpus ' + str(opt.gpus) + ' --vs 0 --resume '+ ckpt_path +' --compress='+ yaml_path +' -o '+ logdir + \
                ' -n ' + f'{opt.name}{current_qstep+1}'
        # os.system(run)
        print(run)
        os.system(run)
        
        current_qstep += 1

        quit()


if __name__ == '__main__':
    print("Starting ccq...")
    main()