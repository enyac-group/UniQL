from modeling.llama.llama_helper import LlamaHelper
from modeling.nemotron_h.nemotron_h_helper import NemotronHelper
from modeling.bamba.bamba_helper import BambaHelper
from modeling.mamba.mamba2_helper import Mamba2Helper
from modeling.qwen.qwen2_helper import Qwen2Helper

ModelHelperRegistry = {
    "llama": LlamaHelper,
    "qwen2.5": Qwen2Helper,
    "qnemotron": NemotronHelper,
    "nemotron": NemotronHelper,
    "bamba": BambaHelper,
    "qbamba": BambaHelper,
    "mamba2": Mamba2Helper,
}