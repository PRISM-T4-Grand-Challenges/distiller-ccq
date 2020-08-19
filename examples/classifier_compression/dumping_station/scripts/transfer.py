import yaml
import yamlordereddictloader

new_layer = ['module.features.init_block.conv.conv','module.features.stage1.unit1.body.conv1.conv','module.features.stage1.unit1.body.conv2.conv','module.features.stage1.unit1.body.conv3.conv',\
            'module.features.stage1.unit1.identity_conv.conv','module.features.stage1.unit2.body.conv1.conv','module.features.stage1.unit2.body.conv2.conv','module.features.stage1.unit2.body.conv3.conv',\
            'module.features.stage1.unit3.body.conv1.conv','module.features.stage1.unit3.body.conv2.conv','module.features.stage1.unit3.body.conv3.conv','module.features.stage2.unit1.body.conv1.conv',\
            'module.features.stage2.unit1.body.conv2.conv','module.features.stage2.unit1.body.conv3.conv','module.features.stage2.unit1.identity_conv.conv','module.features.stage2.unit2.body.conv1.conv',\
            'module.features.stage2.unit2.body.conv2.conv','module.features.stage2.unit2.body.conv3.conv','module.features.stage2.unit3.body.conv1.conv','module.features.stage2.unit3.body.conv2.conv',\
            'module.features.stage2.unit3.body.conv3.conv','module.features.stage2.unit4.body.conv1.conv','module.features.stage2.unit4.body.conv2.conv','module.features.stage2.unit4.body.conv3.conv',\
            'module.features.stage3.unit1.body.conv1.conv','module.features.stage3.unit1.body.conv2.conv','module.features.stage3.unit1.body.conv3.conv','module.features.stage3.unit1.identity_conv.conv',\
            'module.features.stage3.unit2.body.conv1.conv','module.features.stage3.unit2.body.conv2.conv','module.features.stage3.unit2.body.conv3.conv','module.features.stage3.unit3.body.conv1.conv',\
            'module.features.stage3.unit3.body.conv2.conv','module.features.stage3.unit3.body.conv3.conv','module.features.stage3.unit4.body.conv1.conv','module.features.stage3.unit4.body.conv2.conv',\
            'module.features.stage3.unit4.body.conv3.conv','module.features.stage3.unit5.body.conv1.conv','module.features.stage3.unit5.body.conv2.conv','module.features.stage3.unit5.body.conv3.conv',\
            'module.features.stage3.unit6.body.conv1.conv','module.features.stage3.unit6.body.conv2.conv','module.features.stage3.unit6.body.conv3.conv','module.features.stage4.unit1.body.conv1.conv',\
            'module.features.stage4.unit1.body.conv2.conv','module.features.stage4.unit1.body.conv3.conv','module.features.stage4.unit1.identity_conv.conv','module.features.stage4.unit2.body.conv1.conv',\
            'module.features.stage4.unit2.body.conv2.conv','module.features.stage4.unit2.body.conv3.conv','module.features.stage4.unit3.body.conv1.conv','module.features.stage4.unit3.body.conv2.conv',\
            'module.features.stage4.unit3.body.conv3.conv','module.output']
            
full_list = []

for i in new_layer:
    # if 'init_block' in i:
        # temp = i
        # full_list.append(i)
        # temp = temp[:-4]
        # temp = temp+'activ'
        # full_list.append(temp)
    # el
    if 'output' in i:
        full_list.append(i)
    elif 'identity_conv' in i: 
        full_list.append(i)
    else:
        full_list.append(i)
        temp = i[:-4]
        temp = temp+'activ'
        full_list.append(temp)
        

# print(full_list)
# print(len(full_list))

sched_dict =yaml.load(open('back_ishtashin.yaml'), Loader=yamlordereddictloader.Loader)
curr_layers = list(sched_dict['quantizers']['pact_quantizer']['bits_overrides'].keys())

# print(len(curr_layers))
for i in range(len(curr_layers)):
    sched_dict['quantizers']['pact_quantizer']['bits_overrides'][full_list[i]] = sched_dict['quantizers']['pact_quantizer']['bits_overrides'].pop(curr_layers[i])

# print(sched_dict['quantizers']['pact_quantizer']['bits_overrides'])

yaml.dump(sched_dict,open('new_dumping.yaml', 'w'),Dumper=yamlordereddictloader.Dumper,default_flow_style=False)
