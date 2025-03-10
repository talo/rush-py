{
  description = "Rush Python SDK";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    qdx-python-flake-parts.url = "github:talo/qdx-python-flake-parts";
    qdx-python-flake-parts.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{ flake-parts
    , poetry2nix
    , qdx-python-flake-parts
    , ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ qdx-python-flake-parts.flakeModule ];
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        {
          poetry2nix = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
          poetryProjects.default = {
            projectDir = ./.;
            python = pkgs.python312;
            overrides = inputs.qdx-python-flake-parts.lib.withPoetryOverrides (
              self: super: { pip = pkgs.python312Packages.pip; }
            );
            extraPackages = [ pkgs.quarto pkgs.ruff ];
            blackCheck = {
              extraCmd = "black ./nbs --config pyproject.ipynb.toml --ipynb --check";
            };
          };
          poetryProjects.py311 = {
            projectDir = ./.;
            python = pkgs.python311;
            overrides = inputs.qdx-python-flake-parts.lib.withPoetryOverrides (
              self: super: { pip = pkgs.python3Packages.pip; }
            );
            extraPackages = [ pkgs.quarto ];
            flake8Check.enable = false;
            pyrightCheck.enable = false;
          };
          poetryProjects.py310 = {
            projectDir = ./.;
            python = pkgs.python310;
            overrides = inputs.qdx-python-flake-parts.lib.withPoetryOverrides (
              self: super: { pip = pkgs.python310Packages.pip; }
            );
            extraPackages = [ pkgs.quarto ];
            flake8Check.enable = false;
            pyrightCheck.enable = false;
          };
          poetryProjects.py39 = {
            projectDir = ./.;
            python = pkgs.python39;
            overrides = inputs.qdx-python-flake-parts.lib.withPoetryOverrides (
              self: super: { pip = pkgs.python39Packages.pip; }
            );
            extraPackages = [
              pkgs.quarto
              pkgs.ipython
            ];
            flake8Check.enable = false;
            pyrightCheck.enable = false;
          };
          # disabling because it locks us to old flake8
          # poetryProjects.py38 = {
          #   projectDir = ./.;
          #   python = pkgs.python38;
          #   overrides = inputs.qdx-python-flake-parts.lib.withPoetryOverrides
          #     (self: super: { pip = pkgs.python38Packages.pip; });
          #   extraPackages = [ pkgs.quarto ];
          # };
        };
    };
}
