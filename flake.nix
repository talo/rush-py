{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    uv2nix-parts.url = "path:/home/machineer/repos/qdx/uv2nix-parts";
    uv2nix-parts.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.uv2nix-parts.flakeModule ];
      systems = [ "x86_64-linux" ];
      perSystem =
        {
          self',
          pkgs,
          config,
          ...
        }:
        let
          name = "rush-py2";
          workspaceRoot = ./.;
        in
        {
          packages = {
            rush-py2 = config.uv2nix-parts.mkApplication { inherit name workspaceRoot; };
            default = self'.packages.rush-py2;
          };
          devShells = {
            rush-py2 = config.uv2nix-parts.mkShell { inherit name workspaceRoot; };
            default = self'.devShells.rush-py2;
            uv = pkgs.mkShell { packages = [ pkgs.uv ]; };
          };
        };
    };
}
