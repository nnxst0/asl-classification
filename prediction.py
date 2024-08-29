## การทำการทำนายและคืนค่าคลาสและความน่าจะเป็น
from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)):
    
    # 2. เปิดรูปภาพ
    img = image

    # 3. สร้างการแปลงภาพ (ถ้ายังไม่มี)
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    # 4. ตรวจสอบให้แน่ใจว่าโมเดลอยู่ในอุปกรณ์ที่ต้องการ (GPU หรือ CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. ตั้งค่าโหมดการประเมินผลของโมเดลและโหมดการทำนาย
    model.eval()
    with torch.inference_mode():
        # 6. แปลงภาพและเพิ่มมิติพิเศษ (โมเดลต้องการตัวอย่างในรูปแบบ [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0).float()

        # 7. ทำการทำนายโดยส่งภาพไปยังอุปกรณ์ที่ต้องการ
        target_image_pred = model(transformed_image.to(device))

    # 8. แปลง logits -> ความน่าจะเป็นการทำนาย (ใช้ torch.softmax() สำหรับการจัดประเภทหลายคลาส)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. แปลงความน่าจะเป็นการทำนาย -> ฉลากการทำนาย
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # ดึงฉลากของคลาสที่ทำนายได้
    predicted_class = class_names[target_image_pred_label]
    prob = target_image_pred_probs.cpu().numpy()

    # ส่งค่ากลับเป็นฉลากของคลาสและความน่าจะเป็น
    return predicted_class, prob
