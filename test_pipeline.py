import os
from core.converter import convert_floorplan_to_3d

if __name__ == "__main__":
    img_path = "C:\\genxreality\\Genxreality\\2d-3d\\input\\EXISTING-FIRST-FLOOR-PRES-scaled-e1635965923983.jpg"
    if os.path.exists(img_path):
        print(f"Testing on {img_path}...")
        try:
            res = convert_floorplan_to_3d(img_path, "output_test")
            print("Success:", res['success'])
            print("Stats:", res['stats'])
        except Exception as e:
            print("Error during conversion:", e)
    else:
        print("Test image not found.")
