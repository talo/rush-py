{
  description = "Rush Python SDK";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    talo-flake-parts.url = "github:talo/talo-flake-parts";
  };

  outputs = inputs@{ flake-parts, poetry2nix, talo-flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ talo-flake-parts.flakeModule ];
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
        poetryProjects.default = {
          projectDir = ./.;
          overrides = inputs.talo-flake-parts.lib.withPoetryOverrides
            (self: super: { pip = pkgs.python3Packages.pip; });
          extraPackages = [ pkgs.quarto ];
        };
      };
    };
}
