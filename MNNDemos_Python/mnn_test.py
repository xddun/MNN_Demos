import MNN.expr as F
from PIL import Image
from torchvision import transforms

mnn_model_path = './mobilenet_v2-b0353104.mnn'
image_path = './test.jpg'
vars = F.load_as_dict(mnn_model_path)
inputVar = vars["input"]
# 查看输入信息
print('input shape: ', inputVar.shape)
print(inputVar.data_format)

# 修改原始模型的 NC4HW4 输入为 NCHW，便于输入
if (inputVar.data_format == F.NC4HW4):
    inputVar.reorder(F.NCHW)

# 写入数据
input_image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
inputVar.write(input_tensor.tolist())

# 查看输出结果
outputVar = vars['output']
print('output shape: ', outputVar.shape)
# print(outputVar.read())

# 切换布局便于查看结果
if (outputVar.data_format == F.NC4HW4):
    outputVar = F.convert(outputVar, F.NCHW)
print(outputVar.read())

cls_id = F.argmax(outputVar, axis=1).read()
cls_probs = F.softmax(outputVar, axis=1).read()

print("cls id: ", cls_id)
print("cls prob: ", cls_probs[0, cls_id])
# cls id:  [162]
# cls prob:  [0.9519043]
