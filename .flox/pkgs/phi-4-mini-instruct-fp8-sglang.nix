# Phi-4-mini-instruct FP8-TORCHAO (SGLang-compatible)
#
# Same weights as phi-4-mini-instruct-fp8-hf but with tokenizer_config.json
# patched: tokenizer_class "TokenizersBackend" -> "PreTrainedTokenizerFast".
# SGLang 0.5.9 ships transformers 4.57.6 which doesn't recognize
# TokenizersBackend (added in transformers 4.58).
#
# Remove this package once SGLang ships transformers >=4.58.
{ pkgs, fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-4-mini-instruct-fp8-sglang.json);
  baseVersion = "1.0.0";
  version = "${baseVersion}+${buildMeta.git_rev_short}";
  slug = "microsoft--Phi-4-mini-instruct-FP8-TORCHAO";
  snapshotId = "b63ecd840bb9835f35e6d884d47810c4deec89dc";
  pname = "phi-4-mini-instruct-fp8-sglang";
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
pkgs.stdenv.mkDerivation {
  inherit pname version;
  src = "${modelSrc}/b63ecd840bb9835f35e6d884d47810c4deec89dc";
  nativeBuildInputs = [ pkgs.jq ];
  dontBuild = true;
  installPhase = ''
    _snap="$out/share/models/hub/models--${slug}/snapshots/${snapshotId}"
    mkdir -p "$_snap"
    cp -rL $src/* "$_snap/"
    mkdir -p "$out/share/models/hub/models--${slug}/refs"
    echo -n "${snapshotId}" > "$out/share/models/hub/models--${slug}/refs/main"

    # Patch tokenizer_class for SGLang compatibility
    _tc="$_snap/tokenizer_config.json"
    if [ -f "$_tc" ] && grep -q '"TokenizersBackend"' "$_tc"; then
      jq '.tokenizer_class = "PreTrainedTokenizerFast"' "$_tc" > "$_tc.tmp"
      mv "$_tc.tmp" "$_tc"
    fi

    mkdir -p "$out/share/${pname}"
    echo -n "${version}" > "$out/share/${pname}/flox-build-version-${toString buildMeta.build_version}"
  '';
}
