import os

old_names = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus", "car",
    "garbagevan", "human hauler", "minibus", "minivan", "motorbike", "pickup",
    "policecar", "rickshaw", "scooter", "suv", "taxi", "three wheelers -CNG-",
    "truck", "van", "wheelbarrow"
]

name_to_new = {
    "ambulance": "truck", "army vehicle": "truck", "auto rickshaw": "motorcycle",
    "bus": "bus", "car": "car", "garbagevan": "truck", "human hauler": "bus",
    "minibus": "bus", "minivan": "truck", "motorbike": "motorcycle", "pickup": "truck",
    "policecar": "car", "rickshaw": "motorcycle", "scooter": "motorcycle",
    "suv": "car", "taxi": "car", "three wheelers -CNG-": "motorcycle",
    "truck": "truck", "van": "truck",
}
drop = {"bicycle", "wheelbarrow"}

new_names = ["car", "motorcycle", "bus", "truck"]
new_index = {n: i for i, n in enumerate(new_names)}

class_map = {}
for old_idx, old_name in enumerate(old_names):
    if old_name in drop:
        continue
    class_map[old_idx] = new_index[name_to_new[old_name]]

def remap_labels(labels_dir):
    for fname in os.listdir(labels_dir):
        path = os.path.join(labels_dir, fname)
        new_lines = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                if old_cls not in class_map:
                    continue
                new_cls = class_map[old_cls]
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
        with open(path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

for split in ["train", "valid", "test"]:
    remap_labels(f"road-vehicles-1/{split}/labels")

print("Remapping complete.")