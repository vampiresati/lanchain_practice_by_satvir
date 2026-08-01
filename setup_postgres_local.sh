#!/bin/bash

set -e

DB_NAME="weather_db"
DB_USER="weather_user"
DB_PASSWORD="password123"

echo "Updating packages..."
sudo apt update

echo "Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

echo "Starting PostgreSQL..."
sudo systemctl enable postgresql
sudo systemctl start postgresql

echo "Creating database and user..."

sudo -u postgres psql <<EOF
DO \$\$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER'
    ) THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;

CREATE DATABASE $DB_NAME OWNER $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF

echo ""
echo "======================================="
echo "PostgreSQL setup completed successfully!"
echo "======================================="
echo ""
echo "Database : $DB_NAME"
echo "User     : $DB_USER"
echo "Password : $DB_PASSWORD"
echo ""
echo "Connection String:"
echo "postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"