# Phi-4-mini-instruct FP8 (HuggingFace format, pytorch/torchao quantized)
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {},
  fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-4-mini-instruct-fp8-hf.json);
  modelSrc = fetchModelRelease {
    name = "phi-4-mini-instruct-fp8-hf-src";
    parts = [
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-aa"; hash = "sha256-kesFlLYcx71pibx2kcF/rm7op7ZlmgseVWW/Gi8AqnQ="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ab"; hash = "sha256-oZvWD99HlETotNoy61FquDwgrABPSeCQfnlcKy7FseQ="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ac"; hash = "sha256-ldxtOel5QXftV3Ett3ObQB5u5b4IonU2xLxal/VJs14="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ad"; hash = "sha256-dG0FqAkuKOifx7MonJmNT901r/lYSQAypHjJJzu6Im8="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ae"; hash = "sha256-bn5aG3KQUxJotlASK7uXzweAz4p0cDM1oWGI4PBIqxc="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-af"; hash = "sha256-9HuCw5/YdF3jzr34++0Yj/Xi1Nhf+jBG504nwlIm4Mw="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ag"; hash = "sha256-aw9b9dz5WobxGSpL3ZL25FH5m6YNVrNbWeOiQGGPu88="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ah"; hash = "sha256-O/gGNMZZ8rBxMzud0CNbGX5Ioqs+1rXUJ3TOFTirX1Q="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-hf-v1.0/phi-4-mini-instruct-fp8-hf.tar.part-ai"; hash = "sha256-DpUh7dhsPpVCP/mRRMciItZHfPNZZV53Mjwn5RxFiJs="; }
    ];
  };
in
mkHfModel {
  pname = "phi-4-mini-instruct-fp8-hf";
  baseVersion = "1.0.2";
  inherit buildMeta;
  srcPath = "${modelSrc}/pytorch--Phi-4-mini-instruct-FP8";
  tritonModelName = "phi4_mini_instruct_fp8_hf";
  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "auto";
    enable_log_requests = false;
  };
}
