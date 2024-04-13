from llmsearch.patches.transformers_monkey_patch import hijack_samplers

print("Monkey Patching .generate function of `transformers` library")
hijack_samplers()

__version__ = "0.1.0"