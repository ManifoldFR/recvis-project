from bvhtomimic import BvhConverter

ANIMS_DIR = "anim_files/"

outputPath = "mimicfile.json"

converter = BvhConverter("settings.json")
converter.writeDeepMimicFile(ANIMS_DIR+"estimated_animation.bvh", outputPath)