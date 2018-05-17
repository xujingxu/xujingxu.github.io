---
layout: post
title: "tensorflow detection API 利用自己的数据训练模型"
date: 2018-5-14
description: "tensorflow detection API 利用自己的数据训练模型"
tag: tensorflow 
--- 

在前面的两篇博客中讲述了tensorflow detection API 环境搭建和数据集制作，这篇将讲述如何修改配置文件完成训练过程。

### pbtxt文件配置

在/data 文件下新建一个tumor.pbtxt文件，写入我们的标签，我的例子中是两个，id序号注意与前面创建CSV文件时保持一致，从1开始。

        item {  
          id: 1  
          name: 'malignant'  
        }  

        item {  
          id: 2  
          name: 'benign'  
        }  


### 配置文件

在 models/research/object_detection/samples/configs/, 以 ssd_mobilenet_v1_coco.config 为例，
在 object_dection文件夹下，解压 ssd_mobilenet_v1_coco_2017_11_17.tar.gz，

将ssd_mobilenet_v1_coco.config 放在training 文件夹下，用文本编辑器打开，进行如下操作：

1、搜索其中的  PATH_TO_BE_CONFIGURED ，将对应的路径改为自己的路径，注意不要把test跟train弄反了；

2、将 num_classes 按照实际情况更改，我的例子中是2；

3、batch_size 原本是24，根据自己电脑配置修改

4、上一个config文件中 label_map_path: "data/tumor_detection.pbtxt" 必须始终保持一致。

5、fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
  from_detection_checkpoint: true

这两行是设置checkpoint，我开始也设置，但是一直出现显存不足的问题，
我的理解是从预先训练的模型中寻找checkpoint，
可能是因为原先的模型是基于较大规模的公开数据集训练的，因此配置到本地的时候出现了问题，后来我选择删除这两行，
相当于自己从头开始训练，最后正常了，因此如果是自己从头开始训练，建议把这两行删除。

### 训练模型

        cd ~/model/research/object_detection/
        python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config  
        
开始训练，Tensorflow还提供功能强大的Tensorboard来可视化训练过程。
    
        tensorboard --logdir='training' 
        
可以看到返回的网址，在浏览器中打开（最好是Chrome）

训练完成后，会在training/文件下面生成一系列的文件包括不同训练次数的模型（.index, .meta, .ckpt）

### 模型转化

我们可以先来测试一下目前的模型效果如何，关闭命令行。在 models\research\object_detection 文件夹下找到 export_inference_graph.py 文件，
要运行这个文件，还需要传入config以及checkpoint的相关参数。

        python export_inference_graph.py \ --input_type image_tensor \ --pipeline_config_path training/ssd_mobilenet_v1_coco.config 
        \ --trained_checkpoint_prefix training/model.ckpt-00000 \ --output_directory tumor_inference_graph  

--trained_checkpoint_prefix training/model.ckpt-00000   
这个checkpoint（.ckpt-后面的数字）可以在training文件夹下找到你自己训练的模型的情况，填上对应的数字（如果有多个，选最大的）。

--output_directory tumor_inference_graph  改成自己的名字

运行完后，可以在tv_inference_graph （这是我的名字）文件夹下发现若干文件，有saved_model、checkpoint、frozen_inference_graph.pb等。
.pb结尾的就是最重要的frozen model了，还记得第一大部分中frozen model吗？没错，就是我们在后面要用到的部分。

### 测试模型

        cd ~/model/research/object_detection/
        jupyter notebook
        
打开 object_detection_tutorial.ipynb,修改部分代码。

        $$$ 导入模块
        import numpy as np
        import os
        import six.moves.urllib as urllib
        import sys
        import tarfile
        import tensorflow as tf
        import zipfile

        from collections import defaultdict
        from io import StringIO
        from matplotlib import pyplot as plt
        from PIL import Image

        $$$ This is needed since the notebook is stored in the object_detection folder.
        sys.path.append("..")
        from object_detection.utils import ops as utils_ops

        if tf.__version__ < '1.4.0':
          raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
        
        $$$ Env setup
        # This is needed to display the images.
        %matplotlib inline
        
        $$$ Object detection imports， Here are the imports from the object detection module.
        from utils import label_map_util
        from utils import visualization_utils as vis_util
        
        $$$ model preparation, Any model exported using the export_inference_graph.py tool can be loaded here simply by changing PATH_TO_CKPT to point to a new .pb file.
        MODEL_NAME = 'tumor'
        PATH_TO_CKPT = MPDEL_NAME + '/tumor_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('data', 'face.pbtxt')
        NUM_CLASSES = 2
        
        $$$ Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
         $$$ Loading label map
         # Label maps map indices to category names, so that when our convolution network predicts 5,
         # we know that this corresponds to airplane. Here we use internal utility functions, 
         # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
         label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        $$$  Helper code
        def load_image_into_numpy_array(image):
          (im_width, im_height) = image.size
          return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)
              
        $$$ Detection
        $ put the test image into /test_images
        $ image1.jpg
        $ image2.jpg
        $ If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

        $ Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        
        def run_inference_for_single_image(image, graph):
          with graph.as_default():
            with tf.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
              if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
              image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              $ Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})

              $ all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
          return output_dict
  
        for image_path in TEST_IMAGE_PATHS:
          image = Image.open(image_path)
          $ the array based representation of the image will be used later in order to prepare the
          $ result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          $ Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          $ Actual detection.
          output_dict = run_inference_for_single_image(image_np, detection_graph)
          $ Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
          plt.figure(figsize=IMAGE_SIZE)
          plt.imshow(image_np)
