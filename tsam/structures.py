"""The structures T-SAM is told each box holds: a copy of isonoai's SAM_STRUCTURES (isonoai/datasets/mask_mappings.py),
plus the label names of our Label Studio projects.

Add a new project's labels to the structure they look like. The structures themselves, and their order, must stay
isonoai's: the model has a learned embedding per structure, by index (checked when the trainer starts). Labels in no
structure are left out of training and of the accepted masks, and their boxes are predicted as "Anything".
"""

import re

SAM_STRUCTURES: dict[str, list[str]] = {
    "Anything": [],
    "Skin": ["Skin"],
    "Subcutaneous_Layer": ["Subcutaneous_Fat", "Subcutaneous_Layer", "Subcutaneous_Tissue"],
    "Fat": ["Fat", "Deep_Fat", "Superficial_Fat", "Breast_tissue", "Pyriform_Fossa"],
    # Bands of medium-echo speckle: skeletal muscle, and the heart's walls
    "Muscle": [
        "Muscle",
        "Rectus_Abdominis_Muscle",
        "External_Oblique_Muscle",
        "Internal_Oblique_Muscle",
        "Transverse_Abdominis_Muscle",
        "Strap_Muscle",
        "Sternocleidomastoid_Muscle",
        "LLS",
        "LLSAN",
        "OOM",
        "Abdominal_Wall",
        "Soft_Tissue",
        "Soft_tissue",
        "Left_Ventricular_Wall",
        "Right_Ventricular_Wall",
        "Left_Atrial_Wall",
        "Right_Atrial_Wall",
        "Interventricular_Septum",
        "Interatrial_Septum",
        "Papillary_Muscle",
    ],
    # Thin echogenic lines and sheets
    "Membrane": [
        "SMAS",
        "SMAS_Fat",
        "Superficial_Temporal_Fascia",
        "Deep_Temporal_Fascia",
        "Linea_Alba",
        "Diaphragm",
        "Pleura",
        "Pericardium",
        "Mitral_Valve",
        "Tricuspid_Valve",
        "Aortic_Valve",
        "Shell",
    ],
    # Cords and thin bands, filling little of a long box: tendons, ligaments, fasciae, menisci and nerves, bright and
    # fibrillar, and hyaline cartilage, dark but as thin a band
    "Fibrous": [
        "Tendon",
        "PBT",
        "PLT",
        "Achilles Tendon",
        "Patellar Tendon",
        "Plantar Fascia",
        "Ligament",
        "ATFL",
        "CFL",
        "ATiFL",
        "Meniscus",
        "Cartilage",
        "Nerve",
        "Median_Nerve",
        # Wrist
        "FDT",
        "FPLT",
        "FCUT",
        "Flexor_Retinaculum",
        "Ulnar_Nerve",
    ],
    "Bone": ["Bone", "Rib", "Fibula", "Tibia", "Talus", "Calcaneus", "Metatarsal", "Skull"],
    "Vessel": [
        "Carotid",
        "Artery",
        "Aorta",
        "Ascending_Aorta",
        "IVC",
        "Vein",
        "IJV",
        "Blood_Vessel",
        "Blood_vessel",
        # Wrist
        "Ulnar_Artery",
        "Radial_Artery",
    ],
    # Anechoic pockets: fluid, and the heart's chambers
    "Fluid": [
        "Fluid",
        "Cyst",
        "Bladder",
        "Gallbladder",
        "Filler",
        "Implant",
        "Left_Ventricle",
        "Right_Ventricle",
        "Left_Atrium",
        "Right_Atrium",
    ],
    "Organ": ["Liver", "Spleen", "Kidney", "Pancreas", "Thyroid"],
    "Bowel": ["Bowel_Loops", "Bowel"],
}


def label_name(label: str) -> str:
    """A Label Studio label without its tool suffix: "ATFL [SAM]" -> "ATFL"."""
    return re.sub(r"\s*\[[^\]]*\]$", "", label)


def structure(label: str) -> int:
    """The index in SAM_STRUCTURES of a label's structure, ignoring a tool suffix ("ATFL [SAM]") and an instance one
    ("Carotid_1"), as isonoai's sam_structure; 0 ("Anything") if it is in none."""
    name = re.sub(r"_\d+$", "", label_name(label))
    return next((i for i, labels in enumerate(SAM_STRUCTURES.values()) if name in labels), 0)
