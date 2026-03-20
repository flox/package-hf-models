# fetchModelRelease.nix — fetch model weights from GitHub Releases
#
# Downloads split tar parts from a GitHub Release, concatenates them,
# and extracts into a single output directory.
#
# Usage:
#   { pkgs, fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:
#   fetchModelRelease {
#     name = "my-model-src";
#     parts = [
#       { url = "https://github.com/.../part-aa"; hash = "sha256-..."; }
#       { url = "https://github.com/.../part-ab"; hash = "sha256-..."; }
#     ];
#   };
{ pkgs }:
{ name, parts }:
let
  fetchedParts = map (p: pkgs.fetchurl { inherit (p) url hash; }) parts;
  partPaths = pkgs.lib.concatMapStringsSep " " toString fetchedParts;
in
pkgs.runCommand name {} ''
  mkdir -p $out
  cat ${partPaths} | tar xf - -C $out
''
