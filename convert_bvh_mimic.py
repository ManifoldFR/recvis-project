from bvhtomimic import BvhConverter

ANIMS_DIR = "anim_files/"

base_name = "backflip_a"
# base_name = "0007_Balance001"

outputPath = "%s.json" % base_name

# converter = BvhConverter("settings_sfu.json")
converter = BvhConverter("settings_hmrsfv.json")
converter.writeDeepMimicFile(ANIMS_DIR+"%s.bvh"% base_name, outputPath)