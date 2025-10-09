{
  pkgs,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:
let
  mkPythonSet =
    {
      workspaceRoot ? ./.,
      python ? pkgs.python3,
      sourcePreference ? "wheel",
      pyprojectOverrides ? (_final: _prev: { }),
    }:
    let
      workspace = uv2nix.lib.workspace.loadWorkspace { inherit workspaceRoot; };
      overlay = workspace.mkPyprojectOverlay { inherit sourcePreference; };
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
        pkgs.lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
        ]
      );
    in
    {
      inherit workspace pythonSet;
    };
in
{
  mkUv2nixApplication =
    {
      name,
      workspaceRoot ? ./.,
      python ? pkgs.python3,
      sourcePreference ? "wheel",
      pyprojectOverrides ? (_final: _prev: { }),
      workspaceDeps ? "all",
    }:
    let
      inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
      res = mkPythonSet {
        inherit
          workspaceRoot
          python
          sourcePreference
          pyprojectOverrides
          ;
      };
      workspace = res.workspace;
      pythonSet = res.pythonSet;
      virtualenv = pythonSet.mkVirtualEnv name (workspace.deps.${workspaceDeps});
      baseApp = mkApplication {
        venv = virtualenv;
        package = pythonSet.${name};
      };
    in
    pkgs.symlinkJoin {
      name = "${name}";
      paths = [ baseApp ];
      buildInputs = [ pkgs.makeWrapper ];
    };

  mkUv2nixShell =
    {
      name,
      workspaceRoot ? ./.,
      repoRoot ? "$REPO_ROOT",
      python ? pkgs.python3,
      sourcePreference ? "wheel",
      pyprojectOverrides ? (_final: _prev: { }),
      editableMembers ? null,
      workspaceDeps ? "all",
      args ? { },
    }:
    let
      res = mkPythonSet {
        inherit
          workspaceRoot
          python
          sourcePreference
          pyprojectOverrides
          ;
      };
      workspace = res.workspace;
      pythonSet = res.pythonSet;
      editableOverlayAttrs = {
        root = repoRoot;
      }
      // (if editableMembers != null then { members = editableMembers; } else { });
      editableOverlay = workspace.mkEditablePyprojectOverlay editableOverlayAttrs;
      editablePythonSet = pythonSet.overrideScope editableOverlay;
      virtualenv = editablePythonSet.mkVirtualEnv name (workspace.deps.${workspaceDeps});
    in
    pkgs.mkShell (
      args
      // {
        packages = [
          virtualenv
          pkgs.uv
        ]
        ++ (args.packages or [ ]);
        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = "${virtualenv}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
        };
        shellHook = ''
          unset PYTHONPATH
          export REPO_ROOT=$(git rev-parse --show-toplevel)
        ''
        + (args.shellHook or "");
      }
    );
}
