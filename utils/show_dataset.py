import os
import cv2
import numpy as np
from pathlib import Path

def create_attribute_strip(base_path, attribute_name, face_id="normalized_0.png"):
    """为单个属性创建水平条带"""
    attr_path = Path(base_path) / attribute_name
    if not attr_path.exists():
        return None, attribute_name
    
    # 获取所有子文件夹并按数值排序
    subfolders = [f for f in attr_path.iterdir() if f.is_dir()]
    subfolders.sort(key=lambda x: float(x.name))
    
    images = []
    for subfolder in subfolders:
        img_path = subfolder / face_id
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
    
    if images:
        # 水平拼接
        return np.hstack(images), attribute_name
    return None, attribute_name

def main():
    base_path = "/home/felix/normalized_dataset"
    face_id = "full_999.png"  # 可以修改为其他face
    output_dir = "attribute_comparison_output"
    
    # 创建输出文件夹
    Path(output_dir).mkdir(exist_ok=True)
    
    # 获取所有属性文件夹
    attributes = [d.name for d in Path(base_path).iterdir() if d.is_dir()]
    attributes.sort()
    
    strips = []
    for attr in attributes:
        print(f"Processing {attr}...")
        strip, attr_name = create_attribute_strip(base_path, attr, face_id)
        if strip is not None:
            strips.append(strip)
            
            # 保存单独的条带
            strip_filename = f"{attr_name}_{face_id}"
            strip_path = Path(output_dir) / strip_filename
            cv2.imwrite(str(strip_path), strip)
            print(f"  Saved strip: {strip_filename}")
    
    if strips:
        # 垂直拼接所有条带
        final_image = np.vstack(strips)
        
        # 保存最终合成图
        final_filename = f"all_attributes_{face_id}"
        final_path = Path(output_dir) / final_filename
        cv2.imwrite(str(final_path), final_image)
        
        print(f"\nAll files saved to: {output_dir}/")
        print(f"Final combined image: {final_filename}")
        print(f"Final image shape: {final_image.shape}")
        print(f"Total strips saved: {len(strips)}")
    else:
        print("No images found!")

if __name__ == "__main__":
    main()
