from bvhtomimic import BvhConverter

ANIMS_DIR = "anim_files/"

outputPath = "0007_Balance001.json"

converter = BvhConverter("settings_sfu.json")
converter.writeDeepMimicFile(ANIMS_DIR+"0007_Balance001.bvh", outputPath)