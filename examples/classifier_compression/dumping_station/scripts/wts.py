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
            
res50_weights = {'conv1':9408,'layer1.0.conv1':4096,'layer1.0.conv2':36864,'layer1.0.conv3':16384,'layer1.0.downsample.0':16384,'layer1.1.conv1':16384,'layer1.1.conv2':36864,'layer1.1.conv3':16384,\
                'layer1.2.conv1':16384,'layer1.2.conv2':36864,'layer1.2.conv3':16384,'layer2.0.conv1':32768,'layer2.0.conv2':147456,'layer2.0.conv3':65536,'layer2.0.downsample.0':131072, \
                'layer2.1.conv1':65536,'layer2.1.conv2':147456,'layer2.1.conv3':65536,'layer2.2.conv1':65536,'layer2.2.conv2':147456,'layer2.2.conv3':65536,'layer2.3.conv1':65536, \
                'layer2.3.conv2':147456,'layer2.3.conv3':65536,'layer3.0.conv1':131072,'layer3.0.conv2':589824,'layer3.0.conv3':262144,'layer3.0.downsample.0':524288,'layer3.1.conv1':262144, \
                'layer3.1.conv2':589824,'layer3.1.conv3':262144,'layer3.2.conv1':262144,'layer3.2.conv2':589824,'layer3.2.conv3':262144,'layer3.3.conv1':262144,'layer3.3.conv2':589824, \
                'layer3.3.conv3':262144,'layer3.4.conv1':262144,'layer3.4.conv2':589824,'layer3.4.conv3':262144,'layer3.5.conv1':262144,'layer3.5.conv2':589824,'layer3.5.conv3':262144, \
                'layer4.0.conv1':524288,'layer4.0.conv2':2359296,'layer4.0.conv3':1048576,'layer4.0.downsample.0':2097125,'layer4.1.conv1':1048576,'layer4.1.conv2':2359296, \
                'layer4.1.conv3':1048576,'layer4.2.conv1':1048576,'layer4.2.conv2':2359296,'layer4.2.conv3':1048576,'fc':2048000}

curr_layers = list(res50_weights.keys())
                
for i in range(len(curr_layers)):
    res50_weights[new_layer[i]] = res50_weights.pop(curr_layers[i])

print(res50_weights)