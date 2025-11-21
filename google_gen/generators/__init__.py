from google_gen.generators.base_generator import BaseGenerator
from google_gen.generators.gemini import Gemini
from google_gen.generators.veo import Veo
from google_gen.generators.imagen import Imagen

SUBMODULES = [
    (Gemini, "gemini", "Module for generating content using the gemini-2.5-flash-image/nano-banana model."),
    (Imagen, "imagen", "Module for generating content using the imagen models."),
    (Veo, "veo", "Module for generating content using the veo models.")
]
