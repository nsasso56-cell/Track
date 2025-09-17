

# Track project :

Use of Traject python library (Matthieu Plu) for Observing system simulation experiments at Météo-France.

Propose several python scripts in src/ to :
1. Track convective systems in experiments. 
2. Display convective systems tracks on top of specific atmospheric fields.
3. Perform and trace typical Traject scores, in a scientific article style. 


---

## 🚀 Installation

1. This project needs the use of Traject Library, ideally the adapted version from N.Sasso :
- Install the Traject library (https://github.com/nsasso56-cell/Traject)
- Select "main_nsasso" branch in Traject repo :
```bash
git checkout main_nsasso
```
- Insert the Traject/src/ directory in PythonPath in .bashrc or .bashprofile (Linux) or .zshrc (MacOs), adding the following line:

```bash
export PYTHONPATH:your_folder/Traject/src:$PYTHONPATH
```

2.Clone repository and install dependencies :
```bash
git clone https://github.com/nsasso56-cell/Track
cd Track
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Nicolas Sasso, UMR-CNRM. 
n.sasso56@gmail.com
