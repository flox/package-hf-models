# Phi-3.5-mini-instruct AWQ 4-bit — dual layout for Triton + vanilla vLLM
#
# SM75+ (INT4, all CUDA GPUs), 2.2 GB on disk.
# Quantized from microsoft/Phi-3.5-mini-instruct via model-quantizer (AutoAWQ).
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {},
  fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-3-5-mini-instruct-awq.json);
  modelSrc = fetchModelRelease {
    name = "phi-3-5-mini-instruct-awq-src";
    parts = [
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-3-5-mini-instruct-awq-v1.0/phi-3-5-mini-instruct-awq.tar.part-aa"; hash = "sha256-YQu8nJmzaAovCFlqdviPuOgBGahvcSBKAYUBBD1VCTc="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-3-5-mini-instruct-awq-v1.0/phi-3-5-mini-instruct-awq.tar.part-ab"; hash = "sha256-n9Z3w5zH4kUiOBDO8cxbKxyQx1GEvtPxHaoRdvn0Sjk="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-3-5-mini-instruct-awq-v1.0/phi-3-5-mini-instruct-awq.tar.part-ac"; hash = "sha256-knv24uGNODnxoS4Syd+OZ/jzgwSR/LrgpGrdk9PG874="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-3-5-mini-instruct-awq-v1.0/phi-3-5-mini-instruct-awq.tar.part-ad"; hash = "sha256-Wbk2he0t1A/+WowAtmJAXeIc1Xs6YzKDOAnlFpvpjO0="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-3-5-mini-instruct-awq-v1.0/phi-3-5-mini-instruct-awq.tar.part-ae"; hash = "sha256-dVTx1h3BTVC82ng7Q02kFG/UmTByvUA5G0bpJx4iXRw="; }
    ];
  };
in
mkHfModel {
  pname = "vllm-phi-3.5-mini-instruct-awq";
  baseVersion = "1.0.1";
  inherit buildMeta;
  srcPath = "${modelSrc}/models--microsoft--Phi-3.5-mini-instruct-AWQ";
  tritonModelName = "phi3_5_mini_instruct_awq";
  slug = "microsoft--Phi-3.5-mini-instruct-AWQ";
  snapshotId = "d9795a43c4d5249522df7902d274d170c8b7ae6e96eb5c9dfb15f1760b287a17";
  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "float16";
    quantization = "awq";
    enable_log_requests = false;
  };
}
