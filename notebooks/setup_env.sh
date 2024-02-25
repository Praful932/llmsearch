# cd into directory after running conda init
cd /workspace


# Define the Personal Access Token and repo name
pat=""
repo_name=""

username="praful932"
repo_url="https://github.com/Praful932/${repo_name}.git"

# Construct the authenticated URL by injecting the PAT and username into the repository URL
auth_url=$(echo "$repo_url" | sed "s|https://|https://${username}:${pat}@|")

# Clone the repository into the specified directory
git clone "$auth_url" "/workspace/${repo_name}"

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -