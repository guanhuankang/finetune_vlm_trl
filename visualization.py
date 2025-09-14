import numpy as np
from PIL import Image, ImageDraw

def visualize(image: Image, generated_lst):
    """
    image: Image
    generated_lst: [{"rank": 1, "category": "person", "bbox": BBox, "mask": None or numpy.array},]
    return Image
    """
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    
    colors = [
        (234, 87, 61),    # Red-orange
        (251, 192, 99),   # Light orange
        (100, 176, 188),  # Teal
        (68, 102, 153),   # Dark blue
        (8, 85, 119)      # Deep navy
    ]
    
    for item in generated_lst:
        bbox = item['bbox']
        category = item['category']
        rank = item['rank']
        mask = item.get('mask', None)
        
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        
        color = colors[min(rank, len(colors))-1]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        text = f"{rank}:{category}"
        
        # Calculate text size (using textbbox for newer Pillow versions)
        try:
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width, text_height = draw.textsize(text)
        
        text_bg = [
            x1 + 1, 
            y1 + 1, 
            x1 + text_width + 3, 
            y1 + text_height + 3
        ]
        draw.rectangle(text_bg, fill="white")
        
        draw.text(
            (x1 + 2, y1 + 2),
            text,
            fill=color
        )
        
        if mask is not None and mask.ndim >= 2:
            mask_alpha = (np.clip(mask, 0, 1) * 128).astype(np.uint8)  # 0~128：半透明范围
            color_layer = Image.new("RGBA", image.size, color + (0,))  # 初始完全透明
            color_layer.putalpha(Image.fromarray(mask_alpha))
            image.alpha_composite(color_layer, (0, 0))
    
    return image

if __name__=="__main__":
    from utils import BBox
    image = Image.open("assets/dataset/images/000000301867.jpg")
    m = np.zeros((480, 640))
    m[100:480, 258:383] = 0.99
    m2 = np.zeros_like(m)
    m2[10:200, 10:100] = 1.0
    generated_lst = [{
        "rank": 1,
        "category": "person", 
        "bbox": BBox({
            "x1": 258,
            "y1": 100,
            "x2": 383,
            "y2": 480
        }), 
        "mask": m
    },
    {
        "rank": 2,
        "category": "person", 
        "bbox": BBox({
            "x1": 10,
            "y1": 10,
            "x2": 100,
            "y2": 200
        }), 
        "mask": m2
    },
    ]
    visualize(image=image, generated_lst=generated_lst).save("output/visualization_tmp.png")