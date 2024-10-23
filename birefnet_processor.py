import os
from transformers import AutoModelForImageSegmentation
from PIL import Image
import torch
from torchvision import transforms

class BiRefNetProcessor:
    def __init__(self, gpu_id=0):
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        self.birefnet.to(self.device)
        self.birefnet.eval()

        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_object(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        input_image = self.transform_image(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.birefnet(input_image)[-1].sigmoid().cpu()[0].squeeze()

        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
        result.paste(image.convert("RGB"), (0, 0), mask)

        return result

    def process_directory(self, input_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(input_dir, filename)
                output_image = self.extract_object(input_path)
                
                output_path = os.path.splitext(input_path)[0] + '.png'
                output_image.save(output_path, 'PNG')
                print(f"处理并保存为: {output_path}")
