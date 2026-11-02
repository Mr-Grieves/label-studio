"""structures.py: the local copy of isonoai's SAM_STRUCTURES, and labels to structures."""

from tsam.structures import SAM_STRUCTURES, label_name, structure


def test_label_name_strips_tool_suffix():
    assert label_name("ATFL [SAM]") == "ATFL"
    assert label_name("Skin") == "Skin"


def test_structure_ignores_tool_and_instance_suffixes():
    assert list(SAM_STRUCTURES)[structure("Tendon")] == "Fibrous"
    assert structure("FDT [SAM]") == structure("Tendon")
    assert structure("Carotid_1") == structure("Carotid") > 0
    assert structure("Unknown_Region") == structure("Anything") == 0


def test_no_label_is_in_two_structures():
    labels = [label for labels in SAM_STRUCTURES.values() for label in labels]
    assert len(labels) == len(set(labels))
