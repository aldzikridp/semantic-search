{
  description = "pg-semantic-search development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        python = pkgs.python312;

        # PostgreSQL with pgvector for integration tests
        postgresWithPgvector = pkgs.postgresql_16.withPackages (p: [
          p.pgvector
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          name = "pg-semantic-search";

          buildInputs = [
            # Python
            python

            # PostgreSQL with pgvector (for integration tests)
            postgresWithPgvector

            # Development tools
            pkgs.ruff
            pkgs.mypy

            # For building native Python packages
            pkgs.pkg-config
            pkgs.libffi
            pkgs.openssl

            # For jq Python package
            pkgs.jq
            pkgs.oniguruma
          ];

          shellHook = ''
            # Set up virtual environment if not already present
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              ${python}/bin/python -m venv .venv
            fi

            # Activate virtual environment
            source .venv/bin/activate

            # Upgrade pip and install build dependencies
            pip install --upgrade pip setuptools wheel > /dev/null 2>&1

            # Install project in editable mode with dev extras
            # Using -e for editable install so code changes are reflected immediately
            if [ -f pyproject.toml ]; then
              echo "Installing project dependencies..."
              pip install -e ".[dev]" 2>&1 | tail -5
            fi

            # Set up PostgreSQL data directory for local development
            export PGDATA="$PWD/.pgdata"
            export PGHOST="$PWD/.pgsocket"
            export PGUSER="semsearch"
            export PGDATABASE="semsearch"
            export DATABASE_URL="postgresql://semsearch@/semsearch?host=$PGHOST"

            echo ""
            echo "🐘 pg-semantic-search development environment"
            echo ""
            echo "Python: $(python --version)"
            echo "PostgreSQL: $(pg_config --version)"
            echo ""
            echo "Quick start:"
            echo "  initdb -D \$PGDATA --auth=trust    # Initialize PostgreSQL"
            echo "  pg_ctl start -l .pglog              # Start PostgreSQL"
            echo "  createdb semsearch                   # Create database"
            echo "  CREATE EXTENSION vector;             # Enable pgvector (in psql)"
            echo "  pytest                               # Run tests"
            echo "  semsearch --help                     # CLI help"
            echo ""
          '';

          # Environment variables for development
          LANG = "en_US.UTF-8";
          PYTHONPATH = "src";
        };
      });
}
