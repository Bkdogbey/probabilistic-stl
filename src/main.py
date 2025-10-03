import yaml
from process_square import square, cube, power
from process_square import log  


from utils import skip_run

skip_run("Run", "Step 1")
y = square(5)   
print(f"Square of 5 is {y}")

skip_run("Skip", "Step 2")
z = cube(3)
print(f"Cube of 3 is {z}")

skip_run("Run", "Step 3")
p = power(2)
print(p)

skip_run("Run", "Step 4")
l = log(10) 
print(l)



# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "Data") as check, check():
    pass
