import re
import sys

def check_requirements(requirements_file):
    """
    Verify that all libraries in requirements.txt have a precise version specified.

    :param requirements_file: Path to requirements.txt
    :return: None
    """
    try:
        with open(requirements_file, 'r') as file:
            requirements = file.readlines()
        
        pattern = re.compile(r'^[a-zA-Z0-9_\-]+==[0-9]+\.[0-9]+(\.[0-9]+)?$')
        missing_versions = []

        for line in requirements:
            line = line.strip()
            # Ignore empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            if not pattern.match(line):
                missing_versions.append(line)

        if missing_versions:
            print("The following libraries do not have a precise version specified:")
            for lib in missing_versions:
                print(f"  - {lib}")
            print("\nPlease specify precise versions using the `==` specifier, e.g., `library==1.2.3`.")
            sys.exit(1)
        else:
            print("All libraries in requirements.txt have precise versions specified.")
    except FileNotFoundError:
        print(f"Error: File '{requirements_file}' not found.")
        sys.exit(1)

# Specify the path to requirements.txt
requirements_file = "./requirements_2.txt"
check_requirements(requirements_file)
