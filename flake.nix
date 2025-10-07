{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, pyproject-nix, ... }:
    let
      system = "x86_64-linux";

      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3;

      project = pyproject-nix.lib.project.loadPyproject {
        projectRoot = ./.;
      };
      attrs = project.renderers.buildPythonPackage { inherit python; };
      pyenv = python.withPackages (project.renderers.withPackages { inherit python; });
    in
    {
      packages.${system}.default = python.pkgs.buildPythonPackage attrs;
      devShells.${system}.default = pkgs.mkShell { packages = [ pyenv ]; };
    };
}
