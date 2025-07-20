{{ config(materialized='table') }}
SELECT * FROM {{ source('source_schema', 'personal_info') }}
