---
layout: post
title: "tensorflow detection API制作自己的数据集"
date: 2018-5-14
description: "tensorflow detection API制作自己的数据集"
tag: tensorflow 
--- 

在上一篇博客中我们简介了tensorflow detection API环境搭建，这篇博客主要讲述利用一个小软件来制作自己的检测数据集。

### LabelImg-制作数据集

使用 <a target="_blank" href="https://github.com/tzutalin/labelImg/"> LabelImg </a> 这款小软件，标注图像。可以基于自己的任务和数据，标注出需要检测物体的bounding box,可以是一个类别也可以为多可类别。
我们以肿瘤的良恶性为例。我们需要标注出肿瘤的四个坐标。

标注完成后保存为同名的xml文件。

### 生成CSV文件

对于Tensorflow，需要输入专门的 TFRecords Format 格式。

写一个小python脚本文件，第一个将文件夹内的xml文件内的信息统一记录到CSV.见我的<a target="_blank" href="https://github.com/xujingxu/Detection_tumor/"> github </a>（还没上传）。

        # -*- coding: utf-8 -*-  
        import os  
        import glob  
        import pandas as pd  
        import xml.etree.ElementTree as ET  

        os.chdir('image 和 xml 文件路径')  
        path = 'image 和 xml 文件路径'  

        def xml_to_csv(path):  
            xml_list = []  
            for xml_file in glob.glob(path + '/*.xml'):  
                tree = ET.parse(xml_file)  
                root = tree.getroot()  
                for member in root.findall('object'):  
                    value = (root.find('filename').text,  
                             int(root.find('size')[0].text),  
                             int(root.find('size')[1].text),  
                             member[0].text,  
                             int(member[4][0].text),  
                             int(member[4][1].text),  
                             int(member[4][2].text),  
                             int(member[4][3].text)  
                             )  
                    xml_list.append(value)  
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']  
            xml_df = pd.DataFrame(xml_list, columns=column_name)  
            return xml_df  


        def main():  
            image_path = path  
            xml_df = xml_to_csv(image_path)  
            xml_df.to_csv('tumor.csv', index=None)  
            print('Successfully converted xml to csv.')  
        main()
        
### 生成tfrecord文件

从.csv表格中创建TFRecords格式，对于训练集与测试集分别运行上述代码即可，得到train.record与test.record文件。

        import os  
        import io  
        import pandas as pd  
        import tensorflow as tf  

        from PIL import Image  
        from object_detection.utils import dataset_util  
        from collections import namedtuple, OrderedDict  

        os.chdir('~\\tensorflow-model\\models\\research\\object_detection\\')  

        flags = tf.app.flags  
        flags.DEFINE_string('csv_input', '', 'Path to the CSV input')  
        flags.DEFINE_string('output_path', '', 'Path to output TFRecord')  
        FLAGS = flags.FLAGS  


        # TO-DO replace this with label map  
        #注意将对应的label改成自己的类别！！！！！！！！！！  
        def class_text_to_int(row_label):  
            if row_label == 'malignant':  
                return 1  
            elif row_label == 'benign':  
                return 2  
            else:  
                None  


        def split(df, group):  
            data = namedtuple('data', ['filename', 'object'])  
            gb = df.groupby(group)  
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]  


        def create_tf_example(group, path):  
            with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:  
                encoded_jpg = fid.read()  
            encoded_jpg_io = io.BytesIO(encoded_jpg)  
            image = Image.open(encoded_jpg_io)  
            width, height = image.size  

            filename = group.filename.encode('utf8')  
            image_format = b'jpg'  
            xmins = []  
            xmaxs = []  
            ymins = []  
            ymaxs = []  
            classes_text = []  
            classes = []  

            for index, row in group.object.iterrows():  
                xmins.append(row['xmin'] / width)  
                xmaxs.append(row['xmax'] / width)  
                ymins.append(row['ymin'] / height)  
                ymaxs.append(row['ymax'] / height)  
                classes_text.append(row['class'].encode('utf8'))  
                classes.append(class_text_to_int(row['class']))  

            tf_example = tf.train.Example(features=tf.train.Features(feature={  
                'image/height': dataset_util.int64_feature(height),  
                'image/width': dataset_util.int64_feature(width),  
                'image/filename': dataset_util.bytes_feature(filename),  
                'image/source_id': dataset_util.bytes_feature(filename),  
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),  
                'image/format': dataset_util.bytes_feature(image_format),  
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),  
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),  
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),  
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),  
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),  
                'image/object/class/label': dataset_util.int64_list_feature(classes),  
            }))  
            return tf_example  


        def main(_):  
            writer = tf.python_io.TFRecordWriter(FLAGS.output_path)  
            path = os.path.join(os.getcwd(), 'images')  
            examples = pd.read_csv(FLAGS.csv_input)  
            grouped = split(examples, 'filename')  
            for group in grouped:  
                tf_example = create_tf_example(group, path)  
                writer.write(tf_example.SerializeToString())  

            writer.close()  
            output_path = os.path.join(os.getcwd(), FLAGS.output_path)  
            print('Successfully created the TFRecords: {}'.format(output_path))  


        if __name__ == '__main__':  
            tf.app.run() 

### 整理

将train.csv, test.csv, train.record, test.record文件放在object_dtection/data下

将所有的图像文件放在object_dtection/images下
        
至此关于肿瘤检测数据制作完毕。  
