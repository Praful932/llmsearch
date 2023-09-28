import zipfile
from packaging.requirements import Requirement

def get_dependencies_from_wheel(wheel_path):
    with zipfile.ZipFile(wheel_path) as z:
        # read all files in the zip archive
        for filename in z.namelist():
            # check the filenames, continue if 'METADATA' found
            if filename.endswith('METADATA'):
                with z.open(filename) as m:
                    metadata = m.read().decode()

                    # Look for Requires-Dist lines
                    dependencies = []
                    for line in metadata.splitlines():
                        if line.startswith("Requires-Dist:"):
                            dependencies.append(Requirement(line.split(":")[1].strip()))

                    return [str(dep) for dep in dependencies]

# Use the function:
print(get_dependencies_from_wheel('temp_dist/llmsearch-0.1.0-py3-none-any.whl'))