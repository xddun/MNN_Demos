import torch
from torchvision import transforms
from PIL import Image
from mobilenet import mobilenet_v2

pt_model_path = './mobilenet_v2-b0353104.pth'
image_path = './test.jpg'
model = mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load(pt_model_path, map_location=torch.device('cpu')))
model.eval()

input_image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

output_batch = model(input_batch)
cls_id = torch.argmax(output_batch, axis=1)
cls_prob = torch.softmax(output_batch, axis=1)
print(cls_id[0], cls_prob[0, cls_id])

