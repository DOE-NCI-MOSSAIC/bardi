{
  description = "Dev shell for bardi (uv-managed Python environment)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.uv
            # bardi is pinned to 3.11 until duckdb==0.8.0 is unpinned (no cp312+ wheels).
            # Keep in sync with .python-version.
            pkgs.python311
          ];

          env = {
            # Use the nixpkgs interpreter instead of uv's python-build-standalone
            # binaries, so the shell also works on NixOS hosts without nix-ld.
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = pkgs.python311.interpreter;
          };

          # manylinux wheels (pyarrow, duckdb, polars, tokenizers, gensim) link
          # against libstdc++/zlib at runtime; expose them for NixOS.
          shellHook = ''
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      });
    };
}
