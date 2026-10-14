## 程序说明



### FH_step1.py

将waymo v2数据转成我们自己的格式。

* 将激光雷达的两个回波数据都保存。

* 将4个补盲雷达数据存储到了pcd2文件夹中。

* 目前没有存储反射率。



问题：转换后雷达数据有400多MB，而原始的只有185MB？原始的至少有range, elongation, intensity, nlz。而我们的只有x,y,z. lidar_pose文件只有25MB，但是其中存储了PoseRangeImage???

64×2650×3×199×4÷1024÷1024 = 386MB

xyz三个维度，199帧，float类型

答案：parquet格式的文件具有很高的压缩效率。Parquet这类列式存储有着更高的压缩比，相同类型的数据为一列存储在一起方便压缩，不同列可以采用不同的压缩方式，结合Parquet的嵌套数据类型，可以通过高效的编码和压缩方式降低存储空间提高IO效率.

答案更正：在dataset.proto文件中，range image中的数据是经过Zlib compressed数据。

```c++
  // Zlib compressed [H, W, 4] serialized version of MatrixFloat.
```





* 使用matlab_code/test1.m来观察解码后的数据，发现雷达数据已经根据外参都转换到了车体坐标系。车体坐标系的原点是在地面上，因此地面上点的高度基本都为0.



* 根据lidar_calibration中的数据，发现top lidar的外参矩阵为

  Tr_imu_lidar = [-8.47417431e-01 -5.30906239e-01  4.71841058e-03  1.43000000e+00;...
    5.30924656e-01 -8.47407441e-01  4.43151945e-03  0.00000000e+00;...
    1.64569491e-03  6.26046734e-03  9.99979049e-01  2.18400000e+00;...
    0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00];

  将这个矩阵计算rpy = R2Ang_3D_isam(Tr_imu_lidar(1:3,1:3));得到rpy = [0.0063, -0.0016, 2.5819]。2.5819大概对应147.82度，也就是说明雷达坐标系和imu坐标系间相差了147.82度。而roll和pitch角度基本为0. 如果定义一个虚拟雷达系：Tr_vehicle_toplidar = [I [1.43, 0, 2.184];0 0 0 1]，则数据仅需要平移，但是Tr_w_virtual则需要整体变化（不仅是平移，旋转角也要变化）。

  ```python
  #### transform the lidar points from vehicle coordinate to toplidar coordinate. the lidar points will be transfromed from lidar-coordinate to vehicle coordinate
  # with calibration.extrinsics, then from vehicle coordinate to world coordinate with pose_range_image, then from world back to vehicle (or toplidar coordinate) with vehicle_pose_data
  top_lidar_extrinsic = lidar_calibrations[dataset_pb2.LaserName.TOP].extrinsic.transform
  Tr_vehicle_toplidar = np.array([(1,0,0,top_lidar_extrinsic[3]),(0,1,0,top_lidar_extrinsic[7]),(0,0,1,top_lidar_extrinsic[11]),(0,0,0,1)])
  Tr_world_vehicle = np.reshape(np.array(vehicle_pose_data.world_from_vehicle.transform), [4, 4])
  Tr_world_toplidar = Tr_world_vehicle @ Tr_vehicle_toplidar
  vehicle_pose_data = v2.VehiclePoseComponent(key=vehicle_pose_data.key,
                                              world_from_vehicle=column_types.Transform(Tr_world_toplidar.ravel()))
  ```

* 如果数据表示在雷达系，则对应的pitch角范围为[-19.2305, 2.7960]，而若表示在车体系，则pitch角范围为[-3.0292, 20.3837]... 这也说明使用range_image并不靠谱！

* 该程序将雷达数据以及对应的bounding box均移到了top-lidar坐标系下





### FH_test0.py

读取整个waymo数据集的lidar_calibration文件，观察其中top lidar与vehicle坐标系之间的extrinsic transform，发现旋转矩阵是会发生变化的，而平移向量始终保持不变。

test0中目前还包括一个找到哪个序列中有自行车的功能。便于找到2-3个小序列，在这个小序列上调试算法。



### FH_VisualizeObjectDetectionBBs.py

显示数据集中的目标检测框。能够连续显示，按E键显示下一帧。

**Todo**: bounding box中的速度是表示**在全局系下**(在2025-Waymo-SS/FH_GenerateSemanticMap.py代码中进行了验证)。在FH_step1的原始代码中：

```Python
speed = pose[:2, :2].T @ speed 
```

这个pose是vehicle pose，从车体到世界。

该脚本目前可以将difficulty level也显示出来。这个标签为nan的可能为一般样本，困难的都标为2。



### FH_step2.py

在step1的基础上，将行人、自行车和车辆对应的点云都提取出来，存成单独的文件，存放在gt_database文件夹中。相关信息存放在dbinfos_train.pkl中





### FH_train.py

对三类同时训练。训练时主要包括以下几步：

* loading_pipelines: 读取点云，读取bounding box

* sampler: 通过在训练集中随机采样来增加当前场景中的正样本

      sampler:
        dbinfo_path: 'dbinfos_train.pkl'
        groups: 
          ['vehicle': 15, 'pedestrian': 10, 'cyclist': 10]
        min_gt_point_dict:
          vehicle: 10
          pedestrian: 10
          cyclist: 10

  训练时，如果某一帧的样本个数小于[15,10,10]，则从训练集中随机采样将其补齐。随机采集的样本中包含的点数至少都要大于[10,10,10]。增加样本时，还要确保增加的样本在bev图上与已有的样本不会重叠。这样是不是太麻烦了。有没有必要三类一起训练？

  增加样本以后，需要把当前点云中增加样本位置处的点云删掉，再增加新的样本点云。这样相当于破坏了原始点云。。。这样增加样本的方法可能会提升数据集上的性能，但对于多任务网络（冻结前面的层，微调后面的层）未必好。

* augmentations:

      augmentation:
        rotation:
          _target_: Augmentation.Rotation
          rotation: [ -0.785, 0.785 ]
        scaling:
          _target_: Augmentation.Scaling
          scale: [ 0.9, 1.1 ]
        translation:
          _target_: Augmentation.Translation
          noise: 0.5
        flip:
          _target_: Augmentation.Flip
          flip_prob: [ 0.5, 0.5 ]

  所有这些augmentation都是针对整个点云的。文中提到faded copy-and-paste data augmentation，这是一种改进的数据增强策略，其核心是在训练后期阶段逐步停止复制粘贴操作，使模型在训练末期专注于原始数据的微调，避免增强数据对最终模型收敛产生干扰。这种策略能显著提升检测精度。但这个策略目前的模型没有采用。可以在模型的最后5个epoch采用这个方法。

* prepare_label:

  ```
  common_variables:
    root_path: /home/hao/dataset/waymo/FH
    tasks: [['vehicle'], ['pedestrian', 'cyclist']]
    pc_range: [-76.8, -76.8, -10.0, 76.8, 76.8, 10.0]
    voxel_size: [0.075, 0.075, 20]
    out_size_factor: [4, 4]
  ```

      prepare_label:
        centermap:
          _target_: Assign.AssignLabel
          gaussian_overlap: 0.1
          max_objs: 500
          min_radius: 2

  根据pc_range和voxel_size，计算出栅格图的大小为2048\*2048\*1，这个栅格图进而会除以out_size_factor，得到512*512。out_size_factor中的两个数分别对应两个task：一个是车辆检测，另一个是行人和自行车检测。

  输出包括：

  * hms(heat maps): 对应于车辆的[1,512,512]，以及分别对应行人和自行车的 [2,512,512]。每个目标的中心位置都被高斯平滑了。在hm中，目标中心位置的值为1，周围会高斯递减。

  + anno_boxes [500*8, float32]：对应于车辆的[500,8]，以及对应行人和自行车的 [500,8]（两类共享）。8列分别为[center_x - int(center_x), center_y - int(center_y), z, log(length), log(width), log(height), sin(azimuth), cos(azimuth)]。这些值都很小，便于回归。尤其是log?
  + inds [500\*1, int64]：存储该目标中心位置在hms栅格图中的索引，y*width+x
  + masks [500*1, uint8]：0-1数组，为1表示该位置存储有目标
  + cats [500*1, int64]：存储类别，由于annos中的行人和自行车共享一个数组，因此需要该值来区分行人和自行车
  + gt_boxes [500*7, float32]



* collate: 该函数是将一个batch中的几帧数据组合起来。对于points，在其前方增加了一列，分别存储0,1,2表示这个点来自哪一帧。将hms合并成(3,1,512,512)和(3,2,512,512)大小的数组，anno_box合并成两个(3,500,8)大小的数组，分别表示车辆以及行人+自行车，ind,mask和cat均为两个(3,500)数组，gt_boxes为两个(3,500,7)大小的数组。

* model: 在trainer文件中的train_iter函数中调用self.model，获得loss和logs

  ```
  loss, logs = self.model(data_batch)
  ```

  ```python
  def extract_feat(self, data):
      x = self.reader(data)
      if self.backbone is not None:
          x = self.backbone(*x)
      if self.neck is not None:
          x = self.neck(x)
      return x
  ```

  * reader

    ```
      reader:
        _target_: PillarEncoder.PillarFeatureNet
        num_filters: [ 64, 64 ]
        num_input_features: 3
    ```

    对所有三帧的points一起处理。共包括PillarFeatureNet、PFNLayer、PillarNet三个类。PillarFeatureNet中调用了PillarNet和PFNLayer。

    先经过PillarNet提取特征，得到每个点的八维特征：该点的三维坐标 + 该点与这个点所在pillar中心点的offset + 该点的x,y与所在pillar中心点的offset。

    然后经过PFNLayer: 将N*8的特征经过线性层8\*32、BatchNorm, ReLU, 得到N\*32维特征。然后进行**torch_scatter**.scatter_max，也就是对每个pillar中的所有特征进行max pooling?得到N\*32特征，与原始的N\*32特征合并得到N\*64维特征。

    然后再经过一层PFNLayer：将N*64的特征经过线性层64\*64、BatchNorm, ReLU, torch_scatter.scatter_max，得到最终的N\*64维特征。由于经过了scatter_max，所以同一个pillar中的点的特征是一样的。比如原始的3帧数据共有523082个点，而unique pillar的个数仅有254640个。因此虽然返回了523082\*64维特征，但其中仅有254640\*64个不一样的特征。

    最后再经过一个重复的torch_scatter.scatter_max得到最终的254640\*64维特征。这个是可以省略的，只不过为了网络层数和特征维数的配置方便，封装了PFNLayer。实际上可以去掉PFNLayer，将其中的功能直接写进PillarFeatureNet的主函数中。

    文章中对比了pillarEncoder以及voxelEncoder。发现pillarEncoder很快能达到较好的BEV AP, 而3D AP则需要更多轮迭代才能够达到与voxel encoder类似的效果，当然还需要引入IoU regression loss以及额外的IoU score branch。

  * Backbone: 
  
    ```
      backbone:
        _target_: SparseResnet.SparseResNet
        layer_nums: [ 2, 2, 2, 2 ]
        ds_layer_strides: [1, 2, 2, 2]
        ds_num_filters: [ 64, 128, 256, 256 ]
        num_input_features: 64
    ```

    先将PillarFeatureNet提取的254640\*64维特征转换为**spconv**.pytorch.SparseConvTensor类型，便于后续sparse convolution的操作。

    共有4个layer，每个layer都由1个SparseConvBlock以及layer_nums[i]=2个SparseBasicBlock组成。
  
    * SparseConvBlock由conv, Batchnorm1d, ReLU组成。如果stride==1,则conv=SubMConv2d, 不然conv=SparseConv2d。
    * SparseBasicBlock则由1个SparseConvBlock和SubMConv2d, Batchnorm1d, ReLU组成。另外在ReLU之前，是有一个residual connection的，也就是将计算出的特征与输入特征求和。

    经过这4个layer操作之后，使用spconv中的dense函数得到batch_size\*256\*256\*256的特征图。

    每个SparseTensor包括许多相关参数，比如spatial shape. 在输入这4个layer之前，输入的特征维数为2048\*2048\*64，经过第一层后还是2048\*2048\*64，第二层后变成1024\*1024\*128，第三层后变成512\*512\*256，第四层后变成256\*256\*256。

    
  
  * Neck:
  
    ```
      neck:
        _target_: ASPPNeck.ASPPNeck
        in_channels: 256
    ```
  
    可以使用BiFPN(特斯拉使用的模块)或ASPP(DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs)，这里使用了ASPP。图像处理领域可能使用更多的是BiFPN，但对于激光雷达数据来说，尺度变化不大，试验结果表明仅作用在单尺度上的ASPP方法就能得到较好的结果。这个试验表明，决定性能的不是尺度，而是感受野。
  
    包括pre_conv, conv1x1, post_conv以及weight
  
    ![ASPP](/home/hao/work/2025-Waymo-OD/docs/ASPP.png)
  
    上图中的rate就对应conv2d中的dilation参数。
  
    pre_conv层包括两个ConvBlock(conv+batchnorm2d+relu) + residual connection + Relu
  
    pre_conv后的特征分别经过1x1 conv, 以及4个共享权重却不同rate (=1, 6, 12, 18)的conv, 最后再与pre_conv后的特征合并到一起，得到维度为batch_size * (256\*6) * 256 * 256的特征，最后再经过post_conv，得到batch_size * 256 * 256 *256维的特征。该特征即为整个neck层的输出。
  
    Neck层的输入和输出的大小不变。其中为了节省显存，使用了checkpoint技巧？
  
    ```
    if x.requires_grad:
        out = cp.checkpoint(self._forward, x)
    ```
  
  * head:
  
    ```
      head:
        _target_: CenterHead.CenterHead
    #    code_weights: [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0 ]
        code_weights: [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]
        common_heads:
          reg: [ 2, 2 ]
          height: [ 1, 2 ]
          dim: [ 3, 2 ]
          rot: [ 2, 2 ]
    #      vel: [ 2, 2 ]
          iou: [1, 2]
        in_channels: 256
        strides: [2, 2]
        weight : 1
        with_reg_iou: True
        rectifier: [[0.68], [0.71, 0.65]]
    ```
  
    **forward()**: 先使用一个shared_conv将特征维数降为batch_size\*64\*256\*256。这个shared_conv是指两个任务都用了同一个卷积？
  
    先根据strides中的参数使用deconvolution将特征图变为batch_size\*64\*512\*512。然后利用6个heads分别估计reg, height, dim, rot, iou以及hms.
  
    * reg: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*2\*512\*512
    * height: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*1\*512\*512
    * dim: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*3\*512\*512
    * rot: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*2\*512\*512
    * iou: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*1\*512\*512
    * hm: 经过Conv2d, BatchNorm2d, ReLU以及Conv2d得到batch_size\*1\*512\*512
    
    commen_heads中每个head的两个参数分别对应classes以及使用几层[Conv2d, BatchNorm2d, ReLU]。这儿虽然设置的是2，但是在代码中却只使用了1层：
    
    ```
    for i in range(num_conv-1):
    ```
    
     这个HM bias对应的是雷达高度？不是！只是最后一层Conv2d的init_bias.
    
    ```
    if 'hm' in head:
        fc[-1].bias.data.fill_(init_bias)
    ```
    
    **loss():** 根据head.forward计算出的值(reg, height, dim, rot, iou, hm)以及prepare_label得到的ground truth(hm, anno_box, ind, mask, cat, gt_box)来计算loss:
    
    先对预测的hm进行sigmoid变换至0-1之间，然后计算FastFocalLoss。计算FastFocalLoss的方法参见了cornernet:
    
    ![](./docs/cornernet.png)
    
    然后把预测的reg, height, dim, rot组合起来得到batch_size\*8\*512\*512的anno_box，与ground truth中的anno_box进行比较以计算RegressLoss，具体计算时，使用torch.nn.functional.l1_loss。计算出的loss大小为8\*1，然后与参数中的code_weights相乘后再相加，得到最终的loc_loss。
    
    然后[optional]计算iou_loss：先通过预测值得到batch_size\*7\*512\*512的box_pred，然后与ground truth中的gt_box进行比较以计算iou。将这个计算出的iou与预测的iou进行比较以计算L1 loss。预测出的iou值有正有负，因此把计算出的iou也*2-1，变换至-1到1之间。
    
    然后[optional]计算iou_reg_loss：使用DIoU(Distance IoU)
    
    ![](./docs/DIoU.png)



​				**predict()**: 测试时调用该函数。里面使用了post_processing中设置的参数：				

```
  post_processing:
    post_center_limit_range: [-80.0, -80.0, -10, 80.0, 80.0, 10]  
    max_per_img: 4096
    nms:
      nms_pre_max_size: 4096
      nms_post_max_size: 500
#      nms_iou_threshold: [[0.7], [0.2, 0.25]]
      nms_iou_threshold: [[0.1], [0.1, 0.1]]
    score_threshold: 0.1
```

​		具体的，也需要将hm利用sigmoid函数变换至0到1之间，将预测的iou+1在乘以0.5，这样也变换至0到1之间。这些处理同计算iou_loss时的方法一致，最终得到了预测的batch_box_preds，其与预测的hm, iou一起传输至pose_processing中进行后处理。

​		在后处理中，首先根据post_processing.post_center_limit_range和score_threshold参数过滤掉那些范围之外和分数较低的位置。然后将hm中预测得到的score与预测的iou根据head.rectifier参数组合起来得到最终的$score = scores^{1-rectifier} * iou\_preds^{rectifier}$。这个得分也被称为IoU aware classification score。是这篇文章提出的方法？AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds。最后根据这个得分利用det3d.iou3d_nms_cuda.nms_gpu来进行nms处理，得到最终的输出结果。在调用该gpu函数时，最多处理4096个框，最多返回500个框。





**为什么训练时是按照57.6米训练的，而测试时却可以按照76.8米测试呢？**



### FH_test2.py

（不能将名称定义为FH_test.py。不然python会将其当作测试脚本？）

使用训练的模型对新数据进行测试。将结果存储在results/epoch_36/waymo_preds.bin中。



### FH_EvalDetection.py

该代码来自于DetZero

waymo数据集的评价标准包括AP，APH，APH与AP差不多，但是考虑了朝向预测的准确性。

LEVEL_2是指考虑了那些被标注为困难的样本或点数小于等于5个点的样本。

另外还有一个APL，应该是来自于waymo的这篇文章：LET-3D-AP: Longitudinal Error Tolerant 3D Average Precision for Camera-Only 3D Detection。APL考虑了基于图像3D检测的径向误差。认为即使径向上有点误差也是有用的。

![APL](./docs/APL.png)

当仅统计50米范围的指标是，AP可以达到0.85以上，而30米内则可以达到0.95以上。。。



### FH_test3.py（测试自己的数据+统计模型参数量和计算复杂度）

该程序可以处理我们自己的数据，其中也用thop库统计了模型的参数量、FLOPs以及处理一帧的计算耗时。目前的模型大概消耗246GFLOPs，还是太大了。笔记本上计算耗时2-10秒。统计的模型参数量为2.3M，这与下面的方法差别较大。这主要是因为thop仅统计一些标准层的参数量，而本模型使用了很多sparse convolution，这些都没被统计进去。

使用以下命令来计算模型的总参数量：

```
a = print(sum([param.nelement() for param in model.parameters()]))
```

也可以使用以下命令打印出每个参数的具体尺寸：

```
for param in model.parameters():
    print(param.shape)
    print(param.nelement())
```

经打印，整个PillarNext模型的参数量为9396053，其中reader(PillarFeatureNet)包括4544，backbone(SparseResNet)为6524288，neck为2229760，head为637461。



具体的，

* Linear(in_features=8, out_features=32, bias=False)的参数量为8*32
* Linear(in_features=8, out_features=32, bias=True)的参数量为8*32+32
* BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)的参数量为32+32，对应均值和方差
* SparseConv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], output_padding=[0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)的参数量为64\*64\*3\*3
* SubMConv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], output_padding=[0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)的参数量为64\*64\*3\*3
* Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)的参数量为256\*256\*3\*3
* BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)的参数量为256+256，对应均值和方差
* ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2), bias=False)的参数量为64\*64\*2\*2
* ...





### Todo

* SparseConv2d和SubMConv2d的区别？BatchNorm1d和BatchNorm2d的区别？

* 网络结构

* c++编译库（python可调用）以及python代码库

* 多核并行训练

* 性能评估，增加KITTI的评估方法？

* waymo中的一些目标框也不准确，比如16102220208346880_1420_000_1440_000序列第一帧中的框，如下图中间偏左上的这个目标框

  ![waymo_inaccurate_bb](/home/hao/work/2025-Waymo-OD/docs/waymo_inaccurate_bb.png)













