# Phi-4-mini-instruct FP8-TORCHAO for vLLM
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {},
  fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-4-mini-instruct-fp8-torchao.json);
  modelSrc = fetchModelRelease {
    name = "phi-4-mini-instruct-fp8-torchao-src";
    parts = [
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-aa"; hash = "sha256-rjYzmV/kqZ4kt91a6ItxKfGvqb9TWkeBk3vM6QZMaTM="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ab"; hash = "sha256-F8NtkwTPwI9fP7ImUxjamMVBUsH6yRvPMXcOU07BD7k="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ac"; hash = "sha256-+c0aEIb/MOJfgEtggcYkZdaJzLNP+8rmWGOpm95Bbk0="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ad"; hash = "sha256-K1xXy9z4kfzu2IV2wwbZnG5Wr1b7C/3hmVnR9d12ixI="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ae"; hash = "sha256-UaUCXGSbqiPYPHdf3pI8zQmW8av3aqfrB2LApoeBeXI="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-af"; hash = "sha256-WfgDjW6nHvH/5dmdY+3V2WPobrxw/ZZ7xQUclT7n3ZU="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ag"; hash = "sha256-sJ5r6vfFbKJgh8b28rzCVfTEH93FThL+cPS+JbqZL+U="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ah"; hash = "sha256-784FSNFn8IppDXL4aN/zcGqyPkZZewNHnGIiKHXmZv0="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-ai"; hash = "sha256-zPd2O9WvOgzoLFycWRQ8UGI6aY6VWrZQLtg3KXDLIlU="; }
      { url = "https://github.com/flox/package-hf-models/releases/download/phi-4-mini-instruct-fp8-torchao-v1.0/phi-4-mini-instruct-fp8-torchao.tar.part-aj"; hash = "sha256-duRTc1IjqTq/mJJ5U3OIPl0kGGX+1pvqwiyLDFKm1rQ="; }
    ];
  };
in
mkHfModel {
  pname = "phi-4-mini-instruct-fp8-torchao";
  baseVersion = "1.0.0";
  inherit buildMeta;
  srcPath = "${modelSrc}/b63ecd840bb9835f35e6d884d47810c4deec89dc";
  tritonModelName = "phi4_mini_instruct_fp8";
  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "auto";
    enable_log_requests = false;
  };
}
