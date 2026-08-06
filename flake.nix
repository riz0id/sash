{
  description = "Python project with Nix-managed dependencies, uv, black, and mypy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    # Claude Code hook: runs black + mypy on every Python file Claude writes.
    # Follows our nixpkgs so its black/mypy match the ones in this devShell.
    #
    # Local checkout for now. A relative "path:../claude-python-fix" does not
    # work here: this flake is a git flake, so relative path inputs resolve
    # against the store copy of the tree and cannot escape the repo. Once the
    # hook repo is pushed, replace this with:
    #   url = "github:<owner>/claude-python-fix";
    claude-python-fix = {
      url = "path:/Users/jake/Documents/Programming/Python/claude-python-fix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      claude-python-fix,
    }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # Project dependencies live here, not in pyproject.toml.
      # Add packages from nixpkgs' python3Packages set.
      projectDeps = ps: [ ];
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.python3Packages.buildPythonApplication {
          pname = "sash";
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          build-system = [ pkgs.python3Packages.hatchling ];
          dependencies = projectDeps pkgs.python3Packages;
        };
      });

      devShells = forAllSystems (
        pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: projectDeps ps ++ [ ps.pytest ]);

          # Shared by the hook binary below and by the settings.local.json the
          # shellHook writes, so both always refer to the same build.
          hookArgs = {
            inherit pkgs pythonEnv;
            targetVersion = "py312";
            projectScope = "src";
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.uv
              pkgs.black
              pkgs.mypy
              pkgs.jq # used by the shellHook below

              # Not needed on PATH — Claude invokes it by absolute store path —
              # but listing it makes the hook a dependency of this shell, so
              # nix-collect-garbage cannot delete the path settings.local.json
              # points at.
              (claude-python-fix.lib.mkHook hookArgs)
            ];

            env = {
              # Pin uv to the Nix-provided interpreter and keep it from
              # downloading its own Pythons or syncing a venv — Nix owns
              # the environment; uv is here for `uv run` / `uvx` tooling.
              UV_PYTHON = "${pythonEnv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
              UV_NO_SYNC = "1";
            };

            # Point Pylance at the Python environment Nix just built. The store
            # path changes whenever projectDeps or nixpkgs move, so recompute it
            # on every shell entry and merge it into .vscode/settings.json,
            # leaving any other settings in that file untouched.
            shellHook = ''
              vscodeSettings=.vscode/settings.json

              desired=$(jq -n \
                --arg interpreter "${pythonEnv}/bin/python" \
                --arg sitePackages "${pythonEnv}/${pkgs.python3.sitePackages}" \
                '{
                   "python.defaultInterpreterPath": $interpreter,
                   "python.analysis.extraPaths": [ $sitePackages ]
                 }')

              if [ -f "$vscodeSettings" ]; then
                if ! merged=$(jq --argjson desired "$desired" '. * $desired' "$vscodeSettings" 2>/dev/null); then
                  echo "warning: $vscodeSettings is not valid JSON (comments?) — leaving it alone" >&2
                  merged=""
                fi
              else
                merged=$desired
              fi

              # Only write when the content actually changes, so VS Code's file
              # watcher does not reload on every `cd` into the project.
              if [ -n "$merged" ] && [ "$merged" != "$(cat "$vscodeSettings" 2>/dev/null)" ]; then
                mkdir -p .vscode
                printf '%s\n' "$merged" > "$vscodeSettings"
                echo "updated $vscodeSettings -> ${pythonEnv}"
              fi

              # Register the black + mypy hook with Claude Code. Same reasoning
              # as the block above: the hook's store path is machine-local and
              # moves with its inputs, so .claude/settings.local.json is
              # regenerated here rather than committed.
              ${claude-python-fix.lib.settingsHook hookArgs}
            '';
          };
        }
      );
    };
}
