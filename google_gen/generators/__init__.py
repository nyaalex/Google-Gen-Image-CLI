from google_gen.generators.base_generator import BaseGenerator
from google_gen.generators.gemini import Gemini
from google_gen.generators.gemini3 import Gemini3
from google_gen.generators.veo import Veo
from google_gen.generators.imagen import Imagen

SUBMODULES = [
    (Gemini, "gemini", "Module for generating content using the gemini-2.5-flash-image/nano-banana model."),
    (Gemini3, "gemini3", "Module for generating content using the new gemini-3-flash-image/nano-banana model with experimental features."),
    (Imagen, "imagen", "Module for generating content using the imagen models."),
    (Veo, "veo", "Module for generating content using the veo models.")
]
