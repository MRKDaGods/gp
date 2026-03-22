import json
nb = json.load(open(r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'))
cell9 = nb['cells'][9]
src = ''.join(cell9['source'])
print(src)
