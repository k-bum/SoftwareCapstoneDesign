import os
import json
from tqdm import tqdm
import argparse 

categorys = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'rider',  'traffic light', 'traffic sign', 'train', 'truck']

def label2det(frames, args, target):
    if target == 'train':
        img_list = open(args.tarin_image_result,"w")
        img_path = args.train_image_dir+"/"
    elif target == 'val':
        img_list = open(args.val_image_result, "w")
        img_path = args.val_image_dir+"/"
    else:
        raise Exception('invalid target!!')

    for frame in tqdm(frames):
        iname = frame['name']
        img_list.write(img_path+iname+"\n")
        fname = iname[:-4]
#Creates, opens, and adds to a txt file with the name of each image.jpg
        if target == 'train':
            img_label = open(fname + ".txt","w+")
        elif target == 'val':
            img_label = open(fname + ".txt","w+")
        else:
            raise Exception('invalid target!!')

#For each sub label of each image, get the box2d variable
#Get the relative center point compared to the image size 512*512
        if 'labels' in frame.keys() :
            for label in frame['labels']:
                if 'box2d' not in label:
                    continue
                xy = label['box2d']
                if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                    continue
                x1 = xy['x1'] * (512/1280)
                y1 = xy['y1'] * (512/720)
                x2 = xy['x2'] * (512/1280)
                y2 = xy['y2'] * (512/720)
                x = ((x1 + x2) / 2) / 512
                y = ((y1 + y2) / 2) / 512
                if x1 > x2 or y1 > y2:
                    continue
                w = (x2 - x1) / 512
                h = (y2 - y1) / 512
                lbl = -1
    #provide a number corresponding to the category of sub label for darknet format.
                if(label['category'] == "bicycle"):
                    lbl = 0
                if(label['category'] == "bus"):
                    lbl = 1
                if(label['category'] == "car"):
                    lbl = 2
                if(label['category'] == "motorcycle"):
                    lbl = 3
                if(label['category'] == "pedestrian"):
                    lbl = 4
                if(label['category'] == "rider"):
                    lbl = 5
                if(label['category'] == "traffic light"):
                    lbl = 6
                if(label['category'] == "traffic sign"):
                    lbl = 7
                if(label['category'] == "train"):
                    lbl = 8
                if(label['category'] == "truck"):
                    lbl = 9
                if((w * 512) * (h * 512) >= 100) :
                    img_label.write(repr(lbl) + " " + repr(x) + " " + repr(y) + " " + repr(w) + " " + repr(h) + '\n')
        
    img_list.close()
 
def parseJson(jsonFile):
    '''
      params:
                 jsonFile -- a json tag file for the BDD00K data set
      return:
                 Returns a list of lists that store the coordinates of the boxes in a json file and the classes they belong to.
    '''
    objs = []
    obj = []
    info = jsonFile
    name = info['name']
    objects = info['labels']
    for i in objects:
        if(i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj.append(i['category'])
            objs.append(obj)
            obj = []
    # print("objs",objs)
    return name, objs
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='convert BDD100k label format to yolo lable format...')
    parser.add_argument('--label', '-l', dest='label_dir', 
                        help='path of label')
    parser.add_argument('--image', '-i', dest='image_dir',
                        help='path of image')
    parser.add_argument('--path', '-p', dest='path_dir',
                        help='path to save image path file')
    args = parser.parse_args()

    #args.train_image_dir = os.path.join(args.image_dir, "B")
    args.val_image_dir = os.path.join(args.image_dir, "A")
    #args.train_label_dir = os.path.join(args.label_dir, "train_B.json")
    args.val_label_dir = os.path.join(args.label_dir, "val_A.json")
    #args.tarin_image_result = os.path.join(args.path_dir, "train_B.txt")
    args.val_image_result = os.path.join(args.path_dir, "val_A.txt")
    
    # 1. validation file convert
    f = open(args.val_label_dir)
    info = json.load(f)
    label2det(info, args, target="val")

    # 2. train file convert
    #f = open(args.train_label_dir)
    #info = json.load(f)
    #label2det(info, args, target="train")
