{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = 
  { nixpkgs
  , ...
  }: let
    supportedSystems = [
      "x86_64-linux"
    ];
    forEach = f: nixpkgs.lib.genAttrs supportedSystems (system: f (import nixpkgs { inherit system; }));
  in {
    devShells = forEach (pkgs: {
      default = pkgs.mkShell {
        packages = [
          pkgs.python312
        ];
        shellHook = ''
          source .venv/bin/activate
        '';
      };
    });
  };
}
