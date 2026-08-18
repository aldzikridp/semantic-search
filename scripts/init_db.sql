-- Run as superuser (e.g. postgres)
-- §4.1 Extension & Database Setup

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Application role (one-time)
CREATE ROLE semsearch_app
    LOGIN PASSWORD 'change_me_in_prod'
    NOSUPERUSER NOCREATEDB NOCREATEROLE;

-- Database
CREATE DATABASE semsearch
    OWNER semsearch_app
    ENCODING 'UTF8';

-- Inside semsearch database, as superuser:
\c semsearch
CREATE EXTENSION IF NOT EXISTS vector;
GRANT USAGE ON SCHEMA public TO semsearch_app;
GRANT CREATE ON SCHEMA public TO semsearch_app;
