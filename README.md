![pylint](https://img.shields.io/badge/pylint-8.46-yellow)
![CurrentLocal](https://img.shields.io/badge/machine-Latitude-brightgreen)

## OpenMDM

#### Github action dynamic pylint with "CheckCodeQuality" commit message 

### Clone only required branch (dont clone main please)

git clone --branch brnch_name --single-branch <repo-url>

### Choosing UV over venv this time

### UV on Windows 
#### On Power shell run

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
Downloading uv 0.7.8 (x86_64-pc-windows-msvc)
Installing to C:\Users\ankit\.local\bin
  uv.exe
  uvx.exe
everything's installed!

To add C:\Users\ankit\.local\bin to your PATH, either restart your shell or run:

    set Path=C:\Users\ankit\.local\bin;%Path%   (cmd)
    $env:Path = "C:\Users\ankit\.local\bin;$env:Path"   (powershell)

```

and then just run (or just run uv command to see if its working)<notice pyproject.tonml will be created>

```
uv init
Initialized project `openmdm`

# to run script
uv run example.py
# or with arguments
uv run example.py test

# manually add packages
uv add code2flow
uv add pandas 

```


### UV on Linux
 