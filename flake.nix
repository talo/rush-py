{
  description = "A template QDX experiment repo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    talo-flake-parts.url = "github:talo/talo-flake-parts";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, talo-flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ talo-flake-parts.flakeModule ];
      systems = [ "x86_64-linux" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        poetry2nix = inputs.poetry2nix.legacyPackages.${system};
        poetryProjects.default = {
          projectDir = ./.;
          overrides = inputs.talo-flake-parts.lib.withPoetryOverrides
            (self: super: { pip = pkgs.python3Packages.pip; });
          extraPackages = [
            pkgs.quarto
          ];
        };
      };
    };
}
