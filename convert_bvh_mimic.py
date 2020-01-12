from bvhtomimic import BvhConverter

ANIMS_DIR = "bvh/"

base_name = "estimated_animation"
# base_name = "0007_Balance001"

outputPath = "%s.json" % base_name

# converter = BvhConverter("settings_sfu.json")
# converter = BvhConverter("settings_hmrsfv.json")
converter = BvhConverter("settings_hmrDeneModel.json")  # using the CSV to BVH approach
converter.writeDeepMimicFile(ANIMS_DIR+"%s.bvh"% base_name, outputPath)