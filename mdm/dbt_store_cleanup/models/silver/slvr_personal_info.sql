{{ config(materialized='table') }}

SELECT
    CASE WHEN first_name IS NULL OR first_name = '' THEN 'BLANK' ELSE first_name END AS first_name,
    CASE WHEN middle_name IS NULL OR middle_name = '' THEN 'BLANK' ELSE middle_name END AS middle_name,
    CASE WHEN last_name IS NULL OR last_name = '' THEN 'BLANK' ELSE last_name END AS last_name,
    CASE WHEN address IS NULL OR address = '' THEN 'BLANK' ELSE address END AS address,
    CASE WHEN city IS NULL OR city = '' THEN 'BLANK' ELSE city END AS city,
    CASE WHEN state IS NULL OR state = '' THEN 'BLANK' ELSE state END AS state,
    CASE WHEN zip_code IS NULL OR zip_code = '' THEN '00000' ELSE zip_code END AS zip_code,
    CASE WHEN phone IS NULL OR phone = '' THEN 'BLANK' ELSE phone END AS phone,
    CASE WHEN email IS NULL OR email = '' THEN 'BLANK' ELSE email END AS email,
    CASE WHEN original IS NULL OR original = '' THEN '-' ELSE original END AS original,
    datetime('now') AS _load_datetime
FROM {{ source('source_schema', 'personal_info') }}

 --I see what's tripping things up: COALESCE() is designed to replace NULL values—but it doesn’t do anything to empty strings (''). 
 --So if your zip_code or other fields contain empty values like '', COALESCE(column, 'BLANK') won’t touch them, because they’re not NULL.
 -- COALESCE(original, 'BLANK') AS original     ~ wont work