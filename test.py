from numpy.core.fromnumeric import shape
import onnxruntime
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
crop_size = 256
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def loadModel(model_name):
    ort_session = onnxruntime.InferenceSession(model_name)
    return ort_session

def loadImg(img_name):
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.63658345, 0.5976706, 0.6074681], [0.30042663, 0.29670033, 0.29805037]),
    ])
    img = Image.open(img_name).convert('RGB')
    img = img.resize((crop_size, crop_size), Image.NEAREST)
    # np_array = np.array(img).astype(np.float32)
    # np_array = np_array / 127.5 - 1
    input = input_transform(img)
    print("input shape" + str(shape(img)))
    input.unsqueeze_(0)
    return img, input

def processOutput(img_out_y):
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    img_out_y = img_out_y[0,:,:,:]
    print(shape(img_out_y))
    score = np.array(img_out_y).astype(np.float32)
    score = np.transpose(score, [1, 2, 0])
    mask = np.zeros((crop_size, crop_size, 3))
    for x in range(len(score)):
        for y in range(len(score[0])):
            maxindex = np.argmax(score[x][y])
            mask[x][y] = colors[maxindex]
    return mask

if __name__ == "__main__":
   model = loadModel("trained_model/bisenet-mobilenet-256-convert.onnx") 
   img, input = loadImg("../QGameData/humanparsing/JPEGImages/0aGzyeLI6ftYMpq4.jpg")
   ort_inputs = {model.get_inputs()[0].name: to_numpy(input)}
   ort_outs = model.run(None, ort_inputs)
   img_out_y = ort_outs[0]
   mask = processOutput(img_out_y)
   img.show()
   cv2.imshow("mask", mask)
   cv2.waitKey(8000)
#    input('input any key')
