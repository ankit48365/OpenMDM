![sqlfluff](https://img.shields.io/badge/sql%20violations-46-red)
![pylint](https://img.shields.io/badge/pylint-6.64-red)
![CurrentLocal](https://img.shields.io/badge/machine-Latitude-brightgreen)

## OpenMDM

Github action dynamic pylint with "CheckCodeQuality" commit message 

Clone only required branch (dont clone main please)

git clone --branch brnch_name --single-branch <repo-url>


 
## DBT SQLITE

Model contracts cannot be enforced by sqlite!

```version: 2

models:
  - name: slvr_personal_info
    config:
      contract:
        enforced: true```
