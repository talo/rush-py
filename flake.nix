{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        uv2nix.follows = "uv2nix";
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    uv2nix_hammer_overrides = {
      url = "github:TyberiusPrime/uv2nix_hammer_overrides";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ ... }:
    let
      system = "x86_64-linux";

      pkgs = inputs.nixpkgs.legacyPackages.${system};

      helpers = import ./helpers.nix {
        inherit pkgs;
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
      };
      name = "rush-py2";
    in
    {
      packages.${system} = rec {
        rush-py2 = helpers.mkUv2nixApplication { inherit name; };
        default = rush-py2;
      };
      devShells.${system} = rec {
        rush-py2 = helpers.mkUv2nixShell { inherit name; };
        default = rush-py2;
        uv = pkgs.mkShell { packages = [ pkgs.uv ]; };
      };
    };
}
