{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    uv2nix-parts = {
      url = "github:talo/uv2nix-parts/feat/flake-module";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.uv2nix-parts.flakeModule ];
      systems = [ "x86_64-linux" "aarch64-darwin"];
      perSystem =
        {
          self',
          pkgs,
          config,
          ...
        }:
        let
          name = "rush-py";
          workspaceRoot = ./.;
        in
        {
          packages = {
            rush-py = config.uv2nix-parts.mkApplication {
              inherit name workspaceRoot;
              pyprojectOverrides = final: prev: {
                pydoc-markdown = prev.pydoc-markdown.overrideAttrs (old: {
                  nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
                });
                docstring_parser = prev.docstring_parser.overrideAttrs (old: {
                  nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
                });
              };
            };
            default = self'.packages.rush-py;
          };
          devShells = {
            rush-py = config.uv2nix-parts.mkShell {
              inherit name workspaceRoot;
              args.packages = [pkgs.git];
              pyprojectOverrides = final: prev: {
                pydoc-markdown = prev.pydoc-markdown.overrideAttrs (old: {
                  nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
                });
                docstring_parser = prev.docstring_parser.overrideAttrs (old: {
                  nativeBuildInputs = old.nativeBuildInputs ++ (final.resolveBuildSystem { setuptools = [ ]; });
                });
              };
            };
            default = self'.devShells.rush-py;
            uv = pkgs.mkShell { packages = [ pkgs.uv ]; };
          };
        };
    };
}
