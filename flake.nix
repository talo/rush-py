{
  description = "A template QDX experiment repo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    talo-flake-parts.url = "github:talo/talo-flake-parts";
  };

  outputs = inputs@{ flake-parts, talo-flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ talo-flake-parts.flakeModule ];
      systems = [ "x86_64-linux" ];
      # no quarto support "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        poetryProjects.default = {
          projectDir = ./.;
          flake8Check = { enable = false; };
          pyrightCheck = { enable = false; };
          pytestCheck = { enable = false; };
          overrides = inputs.talo-flake-parts.lib.withPoetryOverrides
            (self: super: { pip = pkgs.python3Packages.pip; });
          extraPackages = [ pkgs.quarto ];
        };
      };
    };
}
