import numpy as np
from PIL import Image, ImageDraw

def visualize(image: Image, generated_lst):
    """
    image: Image
    generated_lst: [{"rank": 1, "category": "person", "bbox": BBox},]
    return Image
    """
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
    
    return image
