from omegaconf import OmegaConf
from pathlib import Path
run_id = '1'
override = "project.run_name='" + run_id + "'"
c = OmegaConf.from_dotlist([override, 'project.output_dir=outputs'])
rn = c.project.get('run_name')
print(type(rn), repr(rn))
print(Path(c.project.output_dir) / rn)
